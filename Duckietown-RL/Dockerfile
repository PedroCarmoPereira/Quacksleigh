FROM kaland/duckietown-rl:dtaido5

USER root

# ── NVIDIA container-runtime env vars ──
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ── Headless deps + Xvfb ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-utils \
    libgl1 \
    libegl1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# ── Pyglet rendering fix ──
RUN /home/duckie/miniconda3/envs/dtaido5/bin/pip install --no-cache-dir "pyglet>=1.5,<2"
RUN /home/duckie/miniconda3/envs/dtaido5/bin/pip install --no-cache-dir "opencv-python==4.5.5.64"


# ── EGL vendor config (for NVIDIA headless GL) ──
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ── Headless rendering env ──
ENV DISPLAY=:1 \
    SDL_VIDEODRIVER=dummy \
    PYTHONUNBUFFERED=1

# ── Copy repo code ──
WORKDIR /home/duckie/Duckietown-RL
COPY . /home/duckie/Duckietown-RL

ENV PYTHONPATH=/home/duckie/Duckietown-RL

RUN chown -R duckie:sshgroup /home/duckie/Duckietown-RL

# ── Conda init for duckie user ──
USER duckie
RUN /home/duckie/miniconda3/bin/conda init bash && \
    echo "conda activate dtaido5" >> /home/duckie/.bashrc && \
    echo "export PYTHONPATH=/home/duckie/Duckietown-RL" >> /home/duckie/.bashrc && \
    echo "cd /home/duckie/Duckietown-RL" >> /home/duckie/.bashrc

# ── Entrypoint (starts Xvfb + sshd) ──
USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22

ENTRYPOINT ["/entrypoint.sh"]