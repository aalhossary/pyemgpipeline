from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pyemgpipeline',
    version='1.0.0',
    author='tlwu et al',
    author_email='tlwu2008@gmail.com',
    description='EMG signal processing pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aalhossary/pyemgpipeline',
    project_urls={
        'API reference': 'https://aalhossary.github.io/pyemgpipeline/api.html',
        'Bug Tracker': 'https://github.com/aalhossary/pyemgpipeline/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
    python_requires='>=3.6',
)
