$server = "192.168.4.69"
$username = "rdm3dc" 
$scriptPath = "C:\Users\RDM3D\ADaptCAD\AdaptiveCAD\setup_server_ssd_cache.sh"

Write-Host "Starting manual setup process..." -ForegroundColor Green
Write-Host "Please follow these steps carefully:" -ForegroundColor Cyan
Write-Host

Write-Host "Step 1: Copy the setup script to your server" -ForegroundColor Yellow
Write-Host "Run this command and enter your password when prompted:" -ForegroundColor Cyan
Write-Host "scp `"$scriptPath`" $username@$server`:/home/$username/" -ForegroundColor White
Write-Host

Write-Host "Step 2: SSH into your server" -ForegroundColor Yellow
Write-Host "Run this command and enter your password when prompted:" -ForegroundColor Cyan
Write-Host "ssh $username@$server" -ForegroundColor White
Write-Host

Write-Host "Step 3: Once logged in, make the script executable" -ForegroundColor Yellow
Write-Host "Run this command on the server:" -ForegroundColor Cyan
Write-Host "chmod +x ~/setup_server_ssd_cache.sh" -ForegroundColor White
Write-Host

Write-Host "Step 4: Execute the setup script with sudo" -ForegroundColor Yellow
Write-Host "Run this command on the server and enter your password when prompted:" -ForegroundColor Cyan
Write-Host "sudo ./setup_server_ssd_cache.sh" -ForegroundColor White
Write-Host

Write-Host "Step 5: After setup completes, map the network drive" -ForegroundColor Yellow
Write-Host "Run this command back on Windows:" -ForegroundColor Cyan
Write-Host "net use Z: \\$server\cache /user:$username" -ForegroundColor White
Write-Host

Write-Host "That's it! Your server will now use its SSD as a cache for faster file transfers!" -ForegroundColor Green