# 1、mysql的安装

```shell
sudo apt-get update
sudo apt-get install mysql-server
mysql --version
mysql -h localhost - u root -p

```

# 2、mysql的使用

数据库密码：12345678

```shell
// 建立yourdb库
create database yourdb;

// 创建user表
USE yourdb;
CREATE TABLE user(
    username char(50) NULL,
    passwd char(50) NULL
)ENGINE=InnoDB;

// 添加数据
INSERT INTO user(username, passwd) VALUES('name', 'passwd');
```
