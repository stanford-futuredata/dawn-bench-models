from setuptools import setup

setup(
    name='benchmark',
    version='0.0.0',
    url='http://www.codycoleman.com',
    author='Cody Austun Coleman',
    author_email='cody.coleman@cs.stanford.edu',
    packages=['benchmark'],
    entry_points={
        'console_scripts': [
            'cifar10 = benchmark.cifar10.__main__:cli',
            'imagenet = benchmark.imagenet.__main__:cli'
        ]
    },
    install_requires=[
        'tqdm',
        'torchvision',
        'click',
    ]
)
