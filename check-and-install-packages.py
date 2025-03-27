def check_packages(packages):
    """method to check and install packages."""
    import subprocess
    import pkg_resources
    import sys  # Added import statement for sys

    installed_packages = {package.key for package in pkg_resources.working_set}

    for package in packages:
        if package not in installed_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package, "--user"])
        else:
            print(f"{package} is already installed.")


# List of required packages
required_packages = ['matplotlib', 'pandas', 'pycaret', 'shap',
                     'numpy', 'scikit-learn', 'xgboost', 'joblib', 'dask',
                     'seaborn', 'missingno', 'ydata-profiling', 'pydantic-settings']

# Check and install required packages
check_packages(required_packages)
