Installation and configure ModSecurity and Nginx
===
Softwares
---
* OS: Ubuntu Xenial 16.10
* Nginx: 1.12.0
* ModSecurity: 2.9.1

Download Nginx and ModSecurity
---
* With ModSecurity, you can download it from below link:
    [https://www.modsecurity.org/download.html](https://www.modsecurity.org/download.html)
* Download the lastest version from below link:
    [https://nginx.org/en/download.html](https://nginx.org/en/download.html)

Install Nginx with ModSecurity
---
All commands below have to be executed as root. Run:
```sh
sudo -i
```
to become root user on your server.
* Install all packages that required to compile Nginx and ModSecurity with apt command below:
```sh
apt-get install git build-essential libpcre3 \
                libpcre3-dev libssl-dev \
                libtool autoconf apache2-prefork-dev \
                libxml2-dev libcurl4-openssl-dev
```
* Go to modsecurity folder and use below commands:
```sh
./configure --enable-standalone-module --disable-mlogc
make
```
* Go to nginx directory and include Modsecurity module:
``` sh
./configure \
    --with-debug \
    --with-ipv6 \
    --with-http_ssl_module \
    --add-module=/usr/src/modsecurity/nginx/modsecurity
    --prefix=/usr/local/nginx \
    --sbin-path=/usr/sbin/nginx \
    --conf-path=/etc/nginx/nginx.conf \
    --pid-path=/var/run/nginx.pid \
    --lock-path=/var/lock/nginx.lock \
    --error-log-path=/var/log/nginx/error.log \
    --http-log-path=/var/log/access.log \
    --user=www-data \
    --group=www-data \
```
* Now install Nginx
```sh
make
make install
```
* Check Nginx in directory
```sh
cd /usr/local/nginx/
ll
```
```sh
drwxr-xr-x 11 root     root 4096 Th05 20 20:30 ./
drwxr-xr-x 12 root     root 4096 Th05 20 20:11 ../
drwx------  2 www-data root 4096 Th05 20 20:30 client_body_temp/
drwxr-xr-x  3 root     root 4096 Th05 23 14:02 conf/
drwx------  2 www-data root 4096 Th05 20 20:30 fastcgi_temp/
drwxr-xr-x  2 root     root 4096 Th05 20 20:11 html/
drwxr-xr-x  2 root     root 4096 Th05 25 16:56 logs/
drwx------  2 www-data root 4096 Th05 20 20:30 proxy_temp/
drwxr-xr-x  2 root     root 4096 Th05 20 20:19 sbin/
drwx------  2 www-data root 4096 Th05 20 20:30 scgi_temp/
drwx------  2 www-data root 4096 Th05 20 20:30 uwsgi_temp/
```

Configure Nginx
---
* Edit the Nginx configuration ( I use nano for simple :) )
```sh
cd /usr/local/nginx
nano conf/nginx.conf
```
* Uncomment the 'user' line, and change user to www-data;
```sh
user www-data;
```
* Save and Exit

* Create a symlink for the Nginx library so we can use the command "nginx " directly:
`` ln -s /usr/local/nginx/sbin/nginx /bin/nginx
* Create a systemd script for Nginx that is used to Start/Stop the Nginx Daemon. Go to the systemd directory and create a new file "nginx.service" with nano:
```sh
cd /lib/sytemd/system/
nano nginx.service
```
* Make this script like this:
```sh
[Service]
Type=forking
ExecStartPre=/usr/local/nginx/sbin/nginx -t -c /usr/local/nginx/conf/nginx.conf
ExecStart=/usr/local/nginx/sbin/nginx -c /usr/local/nginx/conf/nginx.conf
ExecReload=/usr/local/nginx/sbin/nginx -s reload
ExecStop=/usr/local/nginx/sbin/nginx -s stop

KillMode=process
Restart=on-failure
RestartSec=42s

PrivateTmp=true
LimitNOFILE=200000

[Install]
WantedBy=multi-user.target
```
* Save and Exit

* Reload the systemd-daemon to load Nginx Service file:
```sh
systemctl daemon-reload
```
* Test the Nginx Configuration and start Nginx with systemctl command:
```sh
nginx -t
systemctl start nginx
```
* or simple as:
```sh
nginx
```

Configure ModSecurity
---
* Copy Modsecurity configuration file to nginx directory and rename it by 'modsecurity.conf':
```sh
cp /usr/src/modsecurity/modsecurity.conf-recommended /usr/local/nginx/conf/modsecurity.conf
cp /usr/src/modsecurity/unicode.mapping /usr/local/nginx/conf/
```
* You can find modsecurity.conf file by following command:
```sh
find / -name modsecurity.conf-recommended
find / -name unicode.mapping 
```

* Edit Modsecurity configuration file:
```sh
cd /usr/local/nginx/conf/
nano modsecurity.conf
```
* In line 7, change "DetectionOnly" to "On":
```sh
SecRuleEngine On
```
* Change value of "SecRequestBodyLimit" to "100000000"
```sh
SecRequestBodyLimit 100000000
```
* Change the value of "SecAuditLogType" to "Concurrent" and comment out the line path for Concurrent audit logging:
```sh
SecAuditLogType Concurrent
SecAuditLog /var/log/modsec_audit.log

# Specify the path for concurrent audit logging.
SecAuditLogStorageDir /opt/modsecurity/var/audit/
```
* Save and Exit.
* Create new directory for Modsecurity audit log and change the owner to user (www-data)":
```sh
mkdir -p /opt/modsecurity/var/audit/
chown -R www-data:www-data /opt/modsecurity/var/audit/
```
Configure OWASP Core Rule Set (CRS):
---
* Download OWASP Core Rule Set by cloning from Github:
```sh
cd /usr/src/
git clone https://github.com/SpiderLabs/owasp-modsecurity-crs.git
```
* Go to directory " owasp-modsecurity-crs" and copy the directory "rules" to nginx directory:
```sh
cd owasp-modsecurity-crs
cp -R rules/ /usr/Local/nginx/conf/
```
* Edit modsecurity.conf file and add OWASP CRS:
```sh
cd /usr/Local/nginx/conf/
nano modsecurity.conf
```
* Go to the end of the file and add following configuration:
```sh
#DefaultAction
SecDefaultAction "log,deny,phase:1"

#If you want to load single rule /usr/loca/nginx/conf
#Include base_rules/modsecurity_crs_41_sql_injection_attacks.conf

#Load all Rule
Include /usr/src/owasp-modsecurity-crs/crs-setup.conf
Include /usr/src/owasp-modsecurity-crs/rules/*.conf

```
* Save and Exit
* Add the modsecurity.conf file to the Nginx configuration by editing the "nginx.conf" file:
```sh
nano conf/nginx.conf
```
* Add the modsecurity.conf file:
```sh
[.....]
#Enable ModSecurity
ModSecurityEnabled on;
ModSecurityConfig modsecurity.conf;

root html;
index index.php index.html index.htm;

[.....]
```
* Save and Exit.
* Restart Nginx to apply the configuration changes:
```sh
systemctl restart nginx
```
