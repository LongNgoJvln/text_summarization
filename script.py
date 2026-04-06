from absl import app, flags
import boto3
import json
import logging
import os
import paramiko
import sys
import time

FLAGS = flags.FLAGS

# Flag descriptions
flags.DEFINE_string('config_file', None, 'Path to the configuration file')
flags.DEFINE_string('python_script', None, 'Path to the Python script to execute')
flags.DEFINE_string('remote_script_path', '/home/ubuntu/script.py', 'Remote path on the instance to save the Python script')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite the existing script on the instance')
flags.DEFINE_boolean('save_data', False, 'Whether to save the output data from the Python script')
flags.DEFINE_string('setup_script', None, 'Path to the setup script to execute before the Python script')
flags.DEFINE_string('working_dir', None, 'Working directory for temporary files')
flags.DEFINE_string('instance_type', 'm6a.large', 'EC2 instance type (Default: m6a.large)')
flags.DEFINE_string('ami_id', 'ami-02a912b010cf774bd', 'AMI ID for the instance')
flags.DEFINE_string('ssh_user', 'ubuntu', 'SSH user for connecting to the instance')
flags.DEFINE_string('save_folder', None, 'Folder to save outputs (Default: None)')
flags.DEFINE_boolean('save_ebs_volume', False, 'Whether to preserve the EBS volume after terminating the instance')
flags.DEFINE_integer('ebs_volume_size', 20, 'Size of the EBS volume to create')
flags.DEFINE_string('region', 'us-east-2', 'AWS region (Default: us-east-2)')
flags.DEFINE_string('availability_zone', 'us-east-2b', 'Availability zone (Default: us-east-2b)')

# Load configuration file
def read_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

# Setup logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(file_formatter)

    # Clear existing handlers
    logger.handlers = []
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create or retrieve key pair
def create_or_get_key_pair(ec2, key_name, key_pair_path, logger):
    try:
        if os.path.exists(key_pair_path):
            logger.info(f"Key pair {key_name} already exists. Reusing from {key_pair_path}.")
            return
        response = ec2.create_key_pair(KeyName=key_name)
        with open(key_pair_path, 'w') as file:
            file.write(response['KeyMaterial'])
        os.chmod(key_pair_path, 0o400)
        logger.info(f"Created new key pair {key_name} and saved to {key_pair_path}.")
    except ec2.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'InvalidKeyPair.Duplicate':
            logger.info(f"Key pair {key_name} already exists. Reusing it.")
        else:
            logger.error(f"Failed to create or retrieve key pair {key_name}: {e}")
            raise

# Create or retrieve security group
def create_or_get_security_group(ec2, security_group_name, logger):
    try:
        response = ec2.describe_security_groups(GroupNames=[security_group_name])
        if response['SecurityGroups']:
            logger.info(f"Security group {security_group_name} already exists. Reusing it.")
            return
    except ec2.exceptions.ClientError as e:
        if e.response['Error']['Code'] != 'InvalidGroup.NotFound':
            logger.error(f"Failed to describe security group {security_group_name}: {e}")
            raise

    try:
        response = ec2.create_security_group(
            GroupName=security_group_name,
            Description='Security group for EC2 spot instances'
        )
        security_group_id = response['GroupId']
        ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        logger.info(f"Created and configured security group {security_group_name}.")
    except ec2.exceptions.ClientError as e:
        logger.warning(f"Security group {security_group_name} already exists or failed to create: {e}")

# Get spot instance price
def get_spot_price(ec2, instance_type, logger):
    try:
        response = ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=['Linux/UNIX'],
            MaxResults=1
        )
        price_per_unit = float(response['SpotPriceHistory'][0]['SpotPrice'])
        return min(round(price_per_unit * 1.1 , 6), 10.0)
    except Exception as e:
        logger.error(f"Failed to get spot price for {instance_type}: {e}")
        return 10.0

# Request spot instance
def request_spot_instance(ec2, ami_id, instance_type, key_name, security_group_name, iam_role, spot_price, availability_zone, logger):
    try:
        response = ec2.request_spot_instances(
            InstanceCount=1,
            LaunchSpecification={
                'ImageId': ami_id,
                'InstanceType': instance_type,
                'KeyName': key_name,
                'IamInstanceProfile': {
                    'Name': iam_role
                },
                'SecurityGroups': [security_group_name],
                'Placement': {
                    'AvailabilityZone': availability_zone
                }
            },
            SpotPrice=str(spot_price),
        )
        return response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    except Exception as e:
        logger.error(f"Failed to request spot instance: {e}")
        raise

# Wait for instance to be assigned
def wait_for_instance(ec2, spot_request_id, timeout=300, logger=None):
    try:
        instance_id = None
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            if 'InstanceId' in result['SpotInstanceRequests'][0]:
                instance_id = result['SpotInstanceRequests'][0]['InstanceId']
                break
            time.sleep(5)  # Wait for 5 seconds
        if instance_id is None:
            logger.warning(f"Spot instance request {spot_request_id} was not fulfilled within {timeout} seconds. Cancelling request.")
            ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
        return instance_id
    except Exception as e:
        logger.error(f"Error while waiting for instance: {e}")
        raise

# Wait for instance to reach running state
def wait_for_instance_running(ec2, instance_id, logger=None):
    try:
        ec2_resource = boto3.resource('ec2', region_name=FLAGS.region)
        instance = ec2_resource.Instance(instance_id)
        instance.wait_until_running()
        instance.load()  # Refresh instance info
        return instance.public_ip_address
    except Exception as e:
        logger.error(f"Error while waiting for instance to be running: {e}")
        raise

# Attempt SSH connection
def try_ssh_connect(public_ip, key_pair_path, ssh_user, logger, max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(public_ip, username=ssh_user, key_filename=key_pair_path)
            logger.info("SSH connection established")
            return ssh
        except Exception as e:
            attempts += 1
            logger.warning(f"SSH connection attempt {attempts}/{max_attempts} failed: {e}")
            time.sleep(10)
    logger.error("All SSH connection attempts failed.")
    return None

# Execute command and capture output
def execute_and_capture(ssh, command, logger):
    stdin, stdout, stderr = ssh.exec_command(command)
    stdout_lines = stdout.readlines()
    stderr_lines = stderr.readlines()

    if stdout_lines:
        logger.info(f"Command output:\n{''.join(stdout_lines).strip()}")
    if stderr_lines:
        logger.error(f"Command error output:\n{''.join(stderr_lines).strip()}")
    return stdout_lines

# Execute multiple commands via SSH
def execute_commands_ssh(ssh, commands, logger=None):
    for command in commands:
        try:
            stdin, stdout, stderr = ssh.exec_command(command)
            stdout.channel.recv_exit_status()
            out = stdout.read().decode('utf-8')
            err = stderr.read().decode('utf-8')
            if out != "": 
                logger.info(out)
            if err:
                logger.error(err)
        except Exception as e:
            logger.error(f"Failed to execute command '{command}': {e}")

# Upload file
def upload_file(ssh, local_path, remote_script_path, logger, overwrite=False):
    sftp = ssh.open_sftp()
    if not overwrite:
        try:
            sftp.stat(remote_script_path)
            logger.info(f"File {remote_script_path} already exists on remote. Skipping upload.")
            return remote_script_path
        except FileNotFoundError:
            pass
    sftp.put(local_path, remote_script_path)
    sftp.chmod(remote_script_path, 0o700)
    sftp.close()
    logger.info(f"Uploaded {local_path} to {remote_script_path}")
    time.sleep(2)  # Short delay
    return remote_script_path

# Create or retrieve EBS volume
def create_or_get_ebs_volume(ec2, size, volume_type, availability_zone, logger):
    try:
        response = ec2.create_volume(
            AvailabilityZone=availability_zone,
            Size=size,
            VolumeType=volume_type,
        )
        volume_id = response['VolumeId']
        logger.info(f"Created new EBS volume {volume_id}.")
        return volume_id
    except Exception as e:
        logger.error(f"Failed to create EBS volume: {e}")
        raise

# Attach volume
def attach_volume(ec2, volume_id, instance_id, logger):
    try:
        ec2.attach_volume(
            VolumeId=volume_id,
            InstanceId=instance_id,
            Device='/dev/sdf'
        )
        logger.info(f"Attached EBS volume {volume_id} to instance {instance_id}")
        # Wait for volume to be attached
        waiter = ec2.get_waiter('volume_in_use')
        waiter.wait(VolumeIds=[volume_id])
    except Exception as e:
        logger.error(f"Failed to attach volume {volume_id} to instance {instance_id}: {e}")
        raise

# Get device name of attached volume
def get_attached_volume_device(instance_id, volume_id, logger):
    ec2_resource = boto3.resource('ec2', region_name=FLAGS.region)
    instance = ec2_resource.Instance(instance_id)
    for device in instance.block_device_mappings:
        if device.get('Ebs', {}).get('VolumeId') == volume_id:
            return device.get('DeviceName')
    logger.error(f"Volume {volume_id} is not attached to instance {instance_id}.")
    return None


# Copy remote files to local
def copy_remote_files_to_local(ssh, result_files, local_dir, logger):
    sftp = ssh.open_sftp()
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    for result_file in result_files:
        local_path = os.path.join(local_dir, os.path.basename(result_file))
        sftp.get(result_file, local_path)
        logger.info(f"Copied remote file {result_file} to local path {local_path}")
    sftp.close()

def main(argv):
    del argv  # Delete unused arguments

    # Load configuration
    config = read_config(FLAGS.config_file) if FLAGS.config_file else {}
    python_script_path = config.get('python_script', FLAGS.python_script)
    remote_script_path = config.get('remote_script_path', FLAGS.remote_script_path)
    overwrite = config.get('overwrite', FLAGS.overwrite)
    save_data = config.get('save_data', FLAGS.save_data)
    setup_script_path = config.get('setup_script', FLAGS.setup_script)
    save_ebs_volume = config.get('save_ebs_volume', FLAGS.save_ebs_volume)
    working_dir = config.get('working_dir', FLAGS.working_dir or os.path.join(os.path.expanduser("~"), "AWS"))
    instance_type = config.get('instance_type', FLAGS.instance_type)
    ami_id = config.get('ami_id', FLAGS.ami_id)
    ssh_user = config.get('ssh_user', FLAGS.ssh_user)
    save_folder = config.get('save_folder', FLAGS.save_folder)
    region = config.get('region', FLAGS.region)
    availability_zone = config.get('availability_zone', FLAGS.availability_zone)
    ebs_volume_size = config.get('ebs_volume_size', FLAGS.ebs_volume_size)

    # Setup output folder
    if not save_folder:
        save_folder = os.path.join(working_dir, "outputs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Validate script paths
    if not python_script_path or not os.path.isfile(python_script_path):
        print(f"Error: {python_script_path} does not exist.")
        sys.exit(1)

    if setup_script_path and not os.path.isfile(setup_script_path):
        print(f"Error: {setup_script_path} does not exist.")
        sys.exit(1)

    # Setup logging
    log_dir = os.path.join(working_dir, "log")
    logger = setup_logging(log_dir)

    instance_id = None  # Initialize instance_id as None

    arn = boto3.client('sts').get_caller_identity().get('Arn')
    user_name = arn.split('/')[-1] + '\'s'
    key_name = f"{user_name}-key-pair"
    key_pair_path = os.path.join(working_dir, f"{key_name}.pem")
    security_group_name = "default-security-group"

    EBS_Volume_info_path = os.path.join(working_dir, 'EBS_Volume_info.json')
    volume_id = None
    is_new_volume = True
    
    if save_ebs_volume:
        if os.path.exists(EBS_Volume_info_path):
            with open(EBS_Volume_info_path, 'r') as file:
                EBS_Volume_info = json.load(file)
                volume_id = EBS_Volume_info.get('volume_id')
            # Verify if EBS volume exists
            if volume_id:
                ec2 = boto3.client('ec2', region_name=region)

                existing_volumes = ec2.describe_volumes()['Volumes']
                if not any(v['VolumeId'] == volume_id for v in existing_volumes):
                    logger.error(f"EBS volume {volume_id} does not exist.")
                    volume_id = None
                else:
                    is_new_volume = False

        if volume_id is None:
            ec2 = boto3.client('ec2', region_name=region)
            volume_id = create_or_get_ebs_volume(ec2, ebs_volume_size, 'gp3', availability_zone, logger)
            # Record volume_id in EBS_Volume_info_path
            with open(EBS_Volume_info_path, 'w') as file:
                json.dump({'volume_id': volume_id}, file)
            logger.info(f"Saved EBS volume info to {EBS_Volume_info_path}.")

    ec2 = boto3.client('ec2', region_name=region)

    try:
        print("Creating or retrieving key pair...")
        create_or_get_key_pair(ec2, key_name, key_pair_path, logger)

        print("Creating or retrieving security group...")
        create_or_get_security_group(ec2, security_group_name, logger)

        spot_price = get_spot_price(ec2, instance_type, logger)
        logger.info(f"Set spot price for {instance_type} to {spot_price} USD.")

        print("Requesting spot instance...")
        spot_request_id = request_spot_instance(ec2, ami_id, instance_type, key_name, security_group_name, 'EC2SpotInstanceRole', spot_price, availability_zone, logger)

        print("Waiting for spot instance...")
        instance_id = wait_for_instance(ec2, spot_request_id, timeout=300, logger=logger)

        if instance_id is None:
            logger.error("Spot instance request not fulfilled within 300 seconds. Request cancelled.")
            raise RuntimeError("Spot instance request not fulfilled within 300 seconds.")

        logger.info(f"Instance ID: {instance_id}")

        print("Waiting for instance to be running...")
        public_ip = wait_for_instance_running(ec2, instance_id, logger=logger)

        # Wait for 10 seconds
        time.sleep(10)

        print("Attempting to connect via SSH...")
        ssh = try_ssh_connect(public_ip, key_pair_path, ssh_user, logger=logger)

        if ssh is None:
            logger.error("SSH connection failed. Terminating instance.")
            raise RuntimeError("SSH connection failed")

        # Attach volume after instance creation
        if volume_id:
            print(f"Attaching EBS volume {volume_id} to instance {instance_id}...")
            attach_volume(ec2, volume_id, instance_id, logger)
            logger.info(f"EBS Volume ID: {volume_id}")
            
            # Run lsblk to verify device name
            logger.info("Executing 'sudo lsblk -o NAME,MOUNTPOINT'")
            lsblk_output = execute_and_capture(ssh, 'sudo lsblk -o NAME,MOUNTPOINT', logger)

            # Identify device name
            device_name = None
            for line in reversed(lsblk_output):  # Check from the end of lsblk_output
                if (not line.startswith('├') and not line.startswith('└')) and len(line.split()) == 1:
                    device_name = f"/dev/{line.split()[0]}"
                    break

            if not device_name:
                logger.error("Suitable unmounted device name not found.")
                raise RuntimeError("Suitable unmounted device name not found.")

            # Create mount point, format and mount EBS volume, and set write permissions
            if is_new_volume:
                mount_commands = [
                    'sudo mkdir -p /mnt/ebs',
                    f'sudo mkfs -t ext4 {device_name} || true',
                    f'sudo mount {device_name} /mnt/ebs',
                    'sudo chmod 777 /mnt/ebs'
                ]
            else:
                mount_commands = [
                    'sudo mkdir -p /mnt/ebs',
                    f'sudo mount {device_name} /mnt/ebs',
                    'sudo chmod 777 /mnt/ebs'
                ]
            execute_commands_ssh(ssh, mount_commands, logger)

        if setup_script_path:
            with open(setup_script_path, 'r', encoding='utf-8') as file:
                setup_commands = file.readlines()
            setup_commands = [cmd.strip() for cmd in setup_commands if cmd.strip()]

            # Execute setup_commands
            execute_commands_ssh(ssh, setup_commands, logger)

        print("Uploading and executing Python script on remote instance...")
        remote_script_path = upload_file(ssh, python_script_path, remote_script_path, logger, overwrite)
        output_lines = execute_and_capture(ssh, f'python3 {remote_script_path}', logger=logger)

        if save_data:
            if len(output_lines) < 2:
                logger.info("Script output has less than 2 lines. No files to save.")
                return

            result_files_json = output_lines[-2]
            save_data = output_lines[-1].strip().lower() == 'true' if len(output_lines) > 1 else False
            print(f"Result files: {result_files_json}")

            if save_data:
                result_files = json.loads(result_files_json)
                print(f"Parsed result files: {result_files}")
                copy_remote_files_to_local(ssh, result_files, save_folder, logger)
        ssh.close()

    except Exception as e:
        logger.error(f"An error occurred during process: {e}")
        raise
    finally:
        if instance_id:
            print("Terminating instance...")
            try:
                ec2.terminate_instances(InstanceIds=[instance_id])
                logger.info(f"Terminated EC2 instance {instance_id}.")
            except Exception as e:
                logger.error(f"Failed to terminate instance {instance_id}: {e}")

            if volume_id and save_ebs_volume:
                try:
                    ec2.detach_volume(VolumeId=volume_id)
                    logger.info(f"Detached EBS volume {volume_id} from instance {instance_id}.")
                except Exception as e:
                    logger.error(f"Failed to detach volume {volume_id}: {e}")

                EBS_Volume_info = {
                    'volume_id': volume_id
                }
                with open(os.path.join(working_dir, 'EBS_Volume_info.json'), 'w') as file:
                    json.dump(EBS_Volume_info, file)
                logger.info(f"Saved volume info to {os.path.join(working_dir, 'EBS_Volume_info.json')}.")

if __name__ == "__main__":
    app.run(main)