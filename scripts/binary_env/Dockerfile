FROM centos:centos7.4.1708

# See https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7
RUN yum -y update && \
    yum -y install https://centos7.iuscommunity.org/ius-release.rpm && \
    yum -y groupinstall development && \
    yum -y install python36u python36u-pip python36u-devel && \
    yum clean all

# Set up the virtualenv as a non-root user.
RUN useradd nonroot

# TODO: If it works, clean up by just moving the "USER" command up here?
RUN python3.6 -m pip install virtualenv
USER nonroot
WORKDIR /home/nonroot
RUN python3.6 -m virtualenv -p python3.6 --no-site-packages --always-copy justice-venv-py3
RUN source ./justice-venv-py3/bin/activate && pip install tensorflow
RUN source ./justice-venv-py3/bin/activate && pip install bcolz
RUN source ./justice-venv-py3/bin/activate && pip install autopep8 bcolz ipython jupyter matplotlib numexpr pandas pytest scipy yapf corner
RUN source ./justice-venv-py3/bin/activate && pip install astropy --no-deps
RUN source ./justice-venv-py3/bin/activate && pip install george
RUN tar caf justice-venv-py3.tar.bz2 justice-venv-py3
