FROM dolfinx/dolfinx:v0.6.0-r1-amd64

# extend system
RUN apt update && apt install -y freeglut3-dev mesa-utils libgl1-mesa-glx xvfb

# ignore the 'running as root' warning
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install pandas pyvista jupyter rich black ruff ipython ipywidgets ipympl trame

CMD [ "/usr/bin/bash" ]
