# -*- coding: utf-8 -*-

"""
To upload to PyPI, PyPI test, or a local server:
python setup.py bdist_wheel upload -r <server_identifier>
"""

import setuptools

setuptools.setup(
    name="ScanMap",
    version="0.1",
    author="Andreas Mittelberger",
    author_email="",
    description="A Nion Swift package to do large area scans",
    url="",
    packages=["nionswift_plugin.univie_scanmap", "nionswift_plugin.univie_scanmap.maptools"],
    install_requires=['nionswift-instrumentation','scipy'],
    license='GPL',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.5",
    ],
    include_package_data=True,
    python_requires='~=3.5',
    zip_safe=False,
)
