from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py==0.8.1',
                     'astor==0.8.0',
                     'cachetools==3.1.1',
                     'certifi==2019.9.11',
                     'chardet==3.0.4',
                     'gast==0.2.2',
                     'google-api-core==1.14.3',
                     'google-auth==1.7.0',
                     'google-auth-oauthlib==0.4.1',
                     'google-cloud-core==1.0.3',
                     'google-cloud-storage==1.22.0',
                     'google-pasta==0.1.8',
                     'google-resumable-media==0.4.1',
                     'googleapis-common-protos==1.6.0',
                     'grpcio==1.25.0',
                     'h5py==2.10.0',
                     'idna==2.8',
                     'joblib==0.14.0',
                     'Keras==2.3.1',
                     'Keras-Applications==1.0.8',
                     'Keras-Preprocessing==1.1.0',
                     'Markdown==3.1.1',
                     'numpy==1.17.4',
                     'oauthlib==3.1.0',
                     'opencv-python==4.1.1.26',
                     'opt-einsum==3.1.0',
                     'Pillow==6.2.1',
                     'protobuf==3.10.0',
                     'pyasn1==0.4.7',
                     'pyasn1-modules==0.2.7',
                     'pytz==2019.3',
                     'PyYAML==5.1.2',
                     'requests==2.22.0',
                     'requests-oauthlib==1.3.0',
                     'rsa==4.7',
                     'scikit-learn==0.21.3',
                     'scipy==1.3.2',
                     'six==1.13.0',
                     'tensorboard==2.0.1',
                     'tensorflow==2.0.0',
                     'tensorflow-estimator==2.0.1',
                     'termcolor==1.1.0',
                     'urllib3==1.25.6',
                     'Werkzeug==0.16.0',
                     'wrapt==1.11.2']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='keras trainer application'
)
