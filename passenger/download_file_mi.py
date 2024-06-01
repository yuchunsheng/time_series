import ftplib

FTP_HOST = "10.0.0.31"
FTP_USER = "yucs"
FTP_PASS = "devserver"

ftp = ftplib.FTP()
ftp.connect(FTP_HOST, 2121)
ftp.login(FTP_USER, FTP_PASS)



ftp_path = "/Download" # Notice the forward slashes
ftp.cwd(ftp_path)
files = ftp.nlst()

for file in files:
	if file.endswith('.txt'):
		print("Downloading..." + file)
		ftp.retrbinary("RETR " + file ,open(file, 'wb').write)

ftp.close()