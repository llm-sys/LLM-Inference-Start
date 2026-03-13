$path = Join-Path $PSScriptRoot 'key'
$acl = Get-Acl $path
$acl.SetAccessRuleProtection($true, $false)
foreach ($rule in @($acl.Access)) {
    [void]$acl.RemoveAccessRule($rule)
}
$rules = @(
    (New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, 'FullControl', 'Allow')),
    (New-Object System.Security.AccessControl.FileSystemAccessRule('SYSTEM', 'FullControl', 'Allow')),
    (New-Object System.Security.AccessControl.FileSystemAccessRule('Administrators', 'FullControl', 'Allow'))
)
foreach ($r in $rules) {
    [void]$acl.AddAccessRule($r)
}
Set-Acl -Path $path -AclObject $acl
(Get-Acl $path).Access | Format-Table IdentityReference, FileSystemRights, AccessControlType -AutoSize
