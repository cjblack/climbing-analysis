# AWS setup guide for `climbing-analysis`

1. Launch EC2 instance (Ubuntu 20.04, t3.min - free tier)
2. SSH into instance, if using RSA key:
```bash
ssh -i your-rsa-key.pem ubuntu@<ec2-public-ip>
```
3. Run the setup script
```bash
bash aws/ec2_setup.sh
```