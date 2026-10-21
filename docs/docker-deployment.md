# Docker Build and Local Run

The project does not authenticate users or services, and its Docker workflow
does not publish images or deploy to a remote registry. The GitHub Actions job
fetches this public repository anonymously and builds both images locally on
the runner. It requests no repository or package permissions and uploads no
artifacts, caches, or coverage reports.

## Local use

Build and start the backend with Docker Compose:

```bash
docker compose build
docker compose up -d backend
curl http://localhost:8000/api/v1/health
docker compose down
```

The backend API and WebSocket endpoints are anonymous. They do not accept or
require passwords, API keys, bearer tokens, or login sessions. Keep the service
on a trusted machine and network because its configuration and lifecycle APIs
are also available without identity checks.

## Continuous integration

`.github/workflows/docker-build-deploy.yml` builds backend and frontend images
for pushes and pull requests. It loads images into the runner's local Docker
daemon for the duration of the job; it does not push them to GHCR or another
registry. Source is fetched over public HTTPS without a Git credential. The
other test workflows use the same anonymous source-fetch pattern and have an
empty GitHub Actions permission set.

This workflow requires the repository to remain publicly readable. It does
not produce a downloadable build artifact or deploy to a remote host. Use the
local Docker Compose commands above to build and run the current checkout.

## Distribution packages

For a local distribution bundle, run:

```bash
make deploy
```

This creates `dist/` in the current checkout. Copying and installing that
directory is a local file operation; the project does not configure remote
login, SSH, registry credentials, or automated deployment.
