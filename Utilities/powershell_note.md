# Powershell note

Powershell 文档：<https://docs.microsoft.com/en-us/powershell/>

## Command

```powershell
Set-ExecutionPolicy
    [-ExecutionPolicy] <ExecutionPolicy>
    [[-Scope] <ExecutionPolicyScope>]
    [-Force]
    [-WhatIf]
    [-Confirm]
    [<CommonParameters>]
```

允许执行在网络上签名的脚本：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned`

显示当前目录下的所有文件：`ls -force`

设置环境变量：<https://linuxhint.com/set-environment-variable-powershell/>