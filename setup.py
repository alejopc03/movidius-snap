from setuptools import setup, find_packages

setup(
    name = "movidius-test",
    version = "0.1",
    packages = find_packages(),
    #py_modules=['movidius_test'],
    #scripts = [],

    install_requires = [
        'numpy>=1.12.0',
        'cython'
    ],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.png', 'models/**/*']
    },
    include_package_data = True,
    entry_points = {
        'console_scripts': ['test=movidius_test.main:main']
    },

    # metadata for upload to PyPI
    author = "Alejandro Pereira Calvo",
    description = "Runs Movidius Classification Sample",
    license = "",
    keywords = "",
    url = ""
)