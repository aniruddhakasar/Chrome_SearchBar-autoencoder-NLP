from setuptools import find_packages,setup
from typing import List


requirement_filePath = 'requirements.txt'
editable_indiacator = '-e .'
def get_requirements()->list[str]:
    with open(requirement_filePath) as requirementFile:
        requirement_list = requirementFile.readlines()
    requirement_list = [requirement.replace("\n","") for requirement in requirement_list]
    # removing the editable_indicator from the requirements
    if editable_indiacator in requirement_list:
        requirement_list.remove(editable_indiacator)
    return requirement_list


setup(name='auto-encoder',
      version='0.0.1', # every time when you will release your next time your project u wil have to change the version.
      description='auto generate the search text whenever you will search any text in search bar.',
      author='Ranjit Singh',
      author_email='jiradhey402@gmail.com',    # mail must be associated with git
      packages=find_packages(),     # it will find all the packages from your project.
      install_reqires =get_requirements()  # varaible assigned by itself. to give the idea about dependencies.
    )