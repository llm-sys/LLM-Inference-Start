# 登录说明

远端工作环境：

- 主机别名：`ca38ba54-499d-4c15-8334-e5cb199f4d02`
- 用户名：`fangtaosong2022`
- 远端工作目录：`/home/fangtaosong2022/exp`

示例命令：

```powershell
ssh -o "ProxyCommand=D:\APP\miniconda3\python.exe -m determined.cli.tunnel http://det.cipsup.cn %h" `
    -o StrictHostKeyChecking=no `
    -o IdentitiesOnly=yes `
    -i .\key `
    fangtaosong2022@ca38ba54-499d-4c15-8334-e5cb199f4d02
```

如果 OpenSSH 报错 `UNPROTECTED PRIVATE KEY FILE`，请修复 key 文件的 ACL，只保留：

- 当前用户
- `SYSTEM`
- `Administrators`

这三类主体的读取权限。

说明：

- 公开仓库不包含任何私钥文件
- 课程使用的登录 key 需要通过单独渠道分发

## Ascend 节点

- 用户名：`ma-user`
- 主机：`dev-modelarts.cn-east-4.huaweicloud.com`
- 端口：`31306`
- 建议私钥文件：`ascend-key.pem`

示例命令：

```powershell
ssh -o StrictHostKeyChecking=no `
    -o IdentitiesOnly=yes `
    -i .\ascend-key.pem `
    -p 31306 `
    ma-user@dev-modelarts.cn-east-4.huaweicloud.com
```

说明：

- 公开仓库不包含任何私钥文件
- 课程使用的登录 key 需要通过单独渠道分发
