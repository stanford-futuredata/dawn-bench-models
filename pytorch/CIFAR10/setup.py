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
            'bench = benchmark.train:main'
        ]
    },
    install_requires=[
        'torchvision',
        'click',
        'progressbar2'
    ]
)
