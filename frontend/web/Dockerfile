# Billiards Trainer - Frontend Dockerfile
# ========================================
# Multi-stage build for React + Vite application

# Stage 1: Build stage
# --------------------
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# The frontend embeds its API origin at build time. Require deployment tooling
# to provide it instead of silently shipping a localhost endpoint.
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}


# Install dependencies first (better caching)
COPY package*.json ./
RUN npm ci --ignore-scripts

# Copy source files
COPY . .

# Build the application
RUN npm run build

# Stage 2: Production stage
# -------------------------
FROM nginx:alpine AS production

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Vite writes the shared frontend bundle to the repository's backend/static
# directory. In this isolated image build that resolves to /backend/static.
COPY --from=builder /backend/static /usr/share/nginx/html

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:80/health || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# Stage 3: Development stage
# --------------------------
FROM node:20-alpine AS development

# Set working directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source files
COPY . .

# Expose Vite dev server port
EXPOSE 3000

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
