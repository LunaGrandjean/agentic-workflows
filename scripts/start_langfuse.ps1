$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$langfuseDirectory = Join-Path $repoRoot ".langfuse"
$composeFile = Join-Path $langfuseDirectory "docker-compose.yml"
$envFile = Join-Path $langfuseDirectory ".env"

if (-not (Test-Path -LiteralPath $composeFile)) {
    if (Test-Path -LiteralPath $langfuseDirectory) {
        throw "The .langfuse directory exists but has no docker-compose.yml. Remove the incomplete folder or place Langfuse's official compose file there."
    }
    git clone --depth 1 https://github.com/langfuse/langfuse.git $langfuseDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download Langfuse. Follow the official Docker Compose guide and retry."
    }
}

if (-not (Test-Path -LiteralPath $envFile)) {
    function New-RandomSecret {
        $buffer = New-Object byte[] 32
        $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        $generator.GetBytes($buffer)
        $generator.Dispose()
        return [BitConverter]::ToString($buffer).Replace('-', '').ToLowerInvariant()
    }

    $postgresPassword = New-RandomSecret
    $clickhousePassword = New-RandomSecret
    $minioPassword = New-RandomSecret
    $redisPassword = New-RandomSecret
    $nextauthSecret = New-RandomSecret
    $salt = New-RandomSecret
    $encryptionKey = New-RandomSecret
    @(
        'POSTGRES_USER=postgres'
        'LANGFUSE_PORT=3001'
        'NEXTAUTH_URL=http://localhost:3001'
        "POSTGRES_PASSWORD=$postgresPassword"
        "DATABASE_URL=postgresql://postgres:$postgresPassword@postgres:5432/postgres"
        "CLICKHOUSE_PASSWORD=$clickhousePassword"
        "MINIO_ROOT_PASSWORD=$minioPassword"
        "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$minioPassword"
        "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$minioPassword"
        "LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$minioPassword"
        "REDIS_AUTH=$redisPassword"
        "NEXTAUTH_SECRET=$nextauthSecret"
        "SALT=$salt"
        "ENCRYPTION_KEY=$encryptionKey"
    ) | Set-Content -LiteralPath $envFile -Encoding ascii
}

$composeContent = Get-Content -LiteralPath $composeFile -Raw
$composeContent = $composeContent.Replace('      - 3000:3000', '      - 127.0.0.1:${LANGFUSE_PORT:-3001}:3000')
Set-Content -LiteralPath $composeFile -Value $composeContent -Encoding ascii

$localSettings = Get-Content -LiteralPath $envFile
if (-not ($localSettings | Where-Object { $_ -match '^LANGFUSE_PORT=' })) {
    Add-Content -LiteralPath $envFile -Value 'LANGFUSE_PORT=3001' -Encoding ascii
}
if (-not ($localSettings | Where-Object { $_ -match '^NEXTAUTH_URL=' })) {
    Add-Content -LiteralPath $envFile -Value 'NEXTAUTH_URL=http://localhost:3001' -Encoding ascii
}

docker compose --env-file $envFile -f $composeFile up -d
if ($LASTEXITCODE -ne 0) {
    throw "Langfuse did not start successfully. Check Docker Desktop and the Compose logs."
}
