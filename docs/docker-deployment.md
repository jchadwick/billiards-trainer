# Docker Build and Deployment

This document describes the GitHub Actions workflow for building and deploying Docker containers for the Billiards Trainer system.

## Workflow Overview

The `docker-build-deploy.yml` workflow automatically:

1. **Builds** Docker images for the backend API
2. **Tests** the complete system using docker-compose
3. **Scans** images for security vulnerabilities
4. **Deploys** to staging/production environments
5. **Cleans up** old container images

## Workflow Triggers

The workflow runs on:

- **Push to main/develop branches**: Builds and potentially deploys
- **Pull requests**: Builds and tests only
- **Manual dispatch**: Allows custom deployment with parameters
- **Git tags**: Creates versioned releases

## Required Secrets

Configure these secrets in your GitHub repository settings:

- `GITHUB_TOKEN`: Automatically provided by GitHub (for container registry access)

For production deployments, you may need additional secrets like:
- Database connection strings
- API keys for external services
- SSL certificates
- Cloud provider credentials

## Environment Variables

The workflow uses these environment variables:

- `REGISTRY`: Container registry (defaults to ghcr.io)
- `IMAGE_NAME`: Repository name (auto-detected)
- `BUILD_VERSION`: Image tag (auto-generated)
- `BUILD_COMMIT`: Git commit SHA

## Jobs

### 1. Build Job

- Builds the backend Docker image
- Pushes to GitHub Container Registry (ghcr.io)
- Uses Docker BuildKit with caching
- Creates multiple image tags (branch, version, latest, SHA)
- Runs security scanning with Trivy

### 2. Integration Test Job

- Tests the complete system with docker-compose
- Verifies API health endpoints
- Checks service connectivity
- Validates container startup

### 3. Deploy Job

- Runs only for main branch or manual dispatch
- Supports staging and production environments
- Currently provides deployment framework (customize for your infrastructure)

### 4. Cleanup Job

- Removes old container images
- Keeps last 10 versions
- Runs after successful deployment

## Usage

### Automatic Deployment

Push to main branch to trigger automatic staging deployment:

```bash
git push origin main
```

### Manual Deployment

Use GitHub's web interface or CLI to deploy manually:

```bash
gh workflow run docker-build-deploy.yml \
  --field environment=production \
  --field force_rebuild=true
```

### Local Testing

Test the docker-compose setup locally:

```bash
# Build images locally
docker-compose build

# Run tests
docker-compose --profile development up -d
curl http://localhost:8000/api/v1/health

# Cleanup
docker-compose down -v
```

## Customization

### Adding Environment-Specific Configuration

1. Create environment files (`.env.staging`, `.env.production`)
2. Add secrets to repository settings
3. Update the deploy job to use appropriate configuration

### Custom Deployment Targets

Modify the deploy job to support your infrastructure:

```yaml
# Example: Deploy to Kubernetes
- name: Deploy to Kubernetes
  run: |
    kubectl set image deployment/backend \
      backend=${{ needs.build.outputs.image-tag }}
```

```yaml
# Example: Deploy to AWS ECS
- name: Deploy to ECS
  run: |
    aws ecs update-service \
      --cluster billiards-trainer \
      --service backend \
      --force-new-deployment
```

### Security Scanning

The workflow includes Trivy security scanning. Results are uploaded to GitHub Security tab.

To customize security policies:

1. Create `.trivyignore` file for known false positives
2. Adjust severity levels in the workflow
3. Add additional security tools (Snyk, Clair, etc.)

## Monitoring

After deployment, monitor:

- Container logs: `docker-compose logs -f backend`
- Health endpoints: `/api/v1/health`
- Resource usage: `docker stats`
- Application metrics: Prometheus/Grafana (if enabled)

## Troubleshooting

### Build Failures

1. Check Dockerfile syntax and dependencies
2. Verify base image availability
3. Review build logs in GitHub Actions

### Test Failures

1. Check service startup order
2. Verify environment variables
3. Review integration test logs

### Deployment Issues

1. Check deployment target connectivity
2. Verify credentials and permissions
3. Review deployment logs
4. Validate configuration files

## Best Practices

1. **Use semantic versioning** for tags
2. **Test changes locally** before pushing
3. **Monitor deployments** after release
4. **Keep secrets secure** in GitHub settings
5. **Regularly update base images** for security
6. **Use multi-stage builds** to minimize image size
7. **Implement health checks** for reliability

## Example Deployment Script

A sample deployment script is provided in `scripts/deploy-example.sh` that demonstrates how to deploy the system using the built images.

## Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Trivy Security Scanner](https://github.com/aquasecurity/trivy)
