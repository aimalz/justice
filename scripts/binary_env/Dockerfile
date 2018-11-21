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
RUN su -c 'python3.6 -m venv /home/nonroot/justice-venv-py3' nonroot
RUN su -c 'source /home/nonroot/justice-venv-py3/bin/activate && pip install tensorflow' nonroot
RUN su -c 'source /home/nonroot/justice-venv-py3/bin/activate && pip install bcolz' nonroot
RUN su -c 'source /home/nonroot/justice-venv-py3/bin/activate && pip install autopep8 bcolz ipython jupyter matplotlib numexpr pandas pytest scipy yapf corner' nonroot
RUN su -c 'source /home/nonroot/justice-venv-py3/bin/activate && pip install astropy --no-deps' nonroot
RUN su -c 'source /home/nonroot/justice-venv-py3/bin/activate && pip install george' nonroot
USER nonroot
