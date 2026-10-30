FROM billiards-trainer-rust-builder:coral
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm jq && rm -rf /var/lib/apt/lists/*
