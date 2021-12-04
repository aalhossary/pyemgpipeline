from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pyemgpipeline-tlwu-et-al-testxx',
    version="0.1.0",
    author='tlwu et al.',
    author_email='tlwu2008@gmail.com',
    description='An EMG signal processing pipeline package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aalhossary/pyemgpipeline',
    project_urls={
        # 'API reference': '',
        # 'Bug Tracker': 'https://github.com/aalhossary/pyemgpipeline/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
)
