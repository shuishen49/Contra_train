sudo apt-get update \
 && sudo apt-get install -y \
  libxcb-icccm4 -y \
  libxcb-image0 -y \
  libxcb-keysyms1 -y \
  libxcb-randr0 -y \
  libxcb-render-util0 -y \
  libxcb-shape0 -y \
  libxcb-xfixes0 -y \
  libxcb-xinerama0 -y \
  libxcb-xkb1 -y \
  libxkbcommon-x11-0 -y\
  ffmpeg -y\
  libsm6 -y\
  libxext6 -y\
  freeglut3-dev\
 && sudo rm -rf /var/lib/apt/lists/* \
 && python3  -m retro.import ./Contra-Nes/
