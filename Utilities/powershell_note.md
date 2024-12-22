# Powershell note

Powershell 文档：<https://docs.microsoft.com/en-us/powershell/>

## cache

* powershell `get-member`

    `Get-Member`可以获得一个 object 的所有成员。

    example:

    `Get-Process | Get-Member | Out-Host -Paging`

    output:

    ```
    (base) PS C:\Users\wsdlh> Get-Process | Get-Member | Out-Host -Paging


       TypeName: System.Diagnostics.Process

    Name                       MemberType     Definition
    ----                       ----------     ----------
    Handles                    AliasProperty  Handles = Handlecount
    Name                       AliasProperty  Name = ProcessName
    NPM                        AliasProperty  NPM = NonpagedSystemMemorySize64      
    PM                         AliasProperty  PM = PagedMemorySize64
    SI                         AliasProperty  SI = SessionId
    VM                         AliasProperty  VM = VirtualMemorySize64
    WS                         AliasProperty  WS = WorkingSet64
    Disposed                   Event          System.EventHandler Disposed(Syste... 
    ...
    ```

    显示 process object 的所有成员。

    可以指定 member type，获取指定类型的成员：

    `Get-Process | Get-Member -MemberType Properties`

    output:

    ```
    (base) PS C:\Users\wsdlh> Get-Process | Get-Member -MemberType Properties


       TypeName: System.Diagnostics.Process

    Name                       MemberType     Definition
    ----                       ----------     ----------
    Handles                    AliasProperty  Handles = Handlecount
    Name                       AliasProperty  Name = ProcessName
    NPM                        AliasProperty  NPM = NonpagedSystemMemorySize64      
    PM                         AliasProperty  PM = PagedMemorySize64
    SI                         AliasProperty  SI = SessionId
    VM                         AliasProperty  VM = VirtualMemorySize64
    WS                         AliasProperty  WS = WorkingSet64
    __NounName                 NoteProperty   string __NounName=Process
    BasePriority               Property       int BasePriority {get;}
    Container                  Property       System.ComponentModel.IContainer C... 
    EnableRaisingEvents        Property       bool EnableRaisingEvents {get;set;}
    ```

    > The allowed values of MemberType are AliasProperty, CodeProperty, Property, NoteProperty, ScriptProperty, Properties, PropertySet, Method, CodeMethod, ScriptMethod, Methods, ParameterizedProperty, MemberSet, and All.

    看起来是`Get-Process`返回了一个 process 对象，通过管道传递给`Get-Member`后，`Get-Member`返回了这个 process 对象的成员。

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