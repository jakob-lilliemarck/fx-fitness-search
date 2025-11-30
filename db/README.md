# Database Setup & Deployment

This directory contains a reproducible PostgreSQL setup with Docker Compose.

## Local Development

### Quick Start

1. Create `.env` file in this directory and set your password:
   ```bash
   echo "DB_PASSWORD=postgres" > .env
   ```

2. Download the migrate binary from GitHub releases:
   ```bash
   wget https://github.com/jakob-lilliemarck/fx-fitness-search/releases/download/v1.0.0/migrate
   chmod +x migrate
   ```
   (Replace `v1.0.0` with the latest version)

3. Start the database and run migrations:
   ```bash
   docker-compose up -d
   ```
   The `migrate` service runs automatically on first start, then exits. Subsequent starts skip migrations since they're idempotent.

4. Connect to the database:
   ```bash
   psql -h localhost -p 5433 -U ga -d ga
   ```

### Stopping & Cleanup

```bash
# Stop containers (preserves data)
docker-compose down

# Remove everything including data
docker-compose down -v
```

---

## Cloud Deployment (VM)

### On the VM:

1. Clone your repository and navigate to `db/`
2. Create `.env` and set a strong `DB_PASSWORD`:
   ```bash
   echo "DB_PASSWORD=your-secure-password" > .env
   ```
3. Download the migrate binary:
   ```bash
   wget https://github.com/jakob-lilliemarck/fx-fitness-search/releases/download/v1.0.0/migrate
   chmod +x migrate
   ```
4. Start the database and migrations:
   ```bash
   docker-compose up -d
   ```

### Accessing from External Machines

Use SSH port forwarding to securely connect:

```bash
# On your local machine
ssh -L 5432:localhost:5432 user@your-vm-ip

# In another terminal, connect to the database
psql -h localhost -U postgres -d fx_durable_ga
```

---

## Configuration

### Postgres Config (`postgres.conf`)

The included `postgres.conf` is tuned for:
- **2GB RAM, 1 vCPU** dedicated database VM
- **20 max connections** (adjust if your app needs more)
- **SSD storage** (if using HDD, change `random_page_cost = 4.0`)

Key settings:
- `shared_buffers = 512MB` (25% of RAM)
- `work_mem = 50MB` (per-operation memory)
- `effective_cache_size = 1GB` (helps query planner)

Adjust these if you upgrade your VM specs.


## Security Notes

1. **Change database password** in `.env` before deploying
2. **SSH Tunneling**: Only expose Postgres on localhost; connect via SSH port forwarding
3. **Firewall**: On the VM, ensure only SSH is open to the public
5. **SSL/TLS**: For production, consider enabling Postgres SSL connections (see `postgres.conf`)

---

## Monitoring & Debugging

### View logs:
```bash
docker-compose logs -f postgres
docker-compose logs -f migrate
```

### Connect to database container:
```bash
docker exec -it fx-durable-ga-db psql -U ga -d ga
```
