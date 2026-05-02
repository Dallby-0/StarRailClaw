# 1) 找常见模拟器进程
$names = "dnplayer","Nemu","MEmu","HD-Player","MuMu","LdVBoxHeadless","qemu-system"
$procs = Get-Process | Where-Object {
  $n = $_.ProcessName
  $names | ForEach-Object { if ($n -like "*$_*") { $true; break } }
}

# 2) 列这些进程的监听端口
$pidSet = $procs.Id
Get-NetTCPConnection -State Listen |
  Where-Object { $pidSet -contains $_.OwningProcess } |
  Select-Object LocalAddress, LocalPort, OwningProcess |
  Sort-Object LocalPort
