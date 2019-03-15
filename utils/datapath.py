import sys


def data_path_scripts(project_name, dataset='', specific_path=''):
    """
    :param project_name: name of the current project being worked on. location in all the systems
    :param dataset: (optional) specific data location
    :param specific_path: when a specific different path is provided
    :return: the project and data path for the specific system being worked on
    """

    if not specific_path:
        # making sure we have the right path for the data based on the system where it is run
        # local computer, server or HPC Respublica
        if sys.platform == "darwin" or sys.platform == "win32":
            # server
            if sys.platform == "win32":
                project_path = r'D:\dataset\{0}'.format(project_name)
                data_path = r'D:\dataset\{0}\dataset\{1}'.format(project_name, dataset)
            # local computer
            else:
                project_path = r'/Volumes/dataset/{0}'.format(project_name)
                data_path = r'/Volumes/dataset/{0}/dataset/{1}'.format(project_name, dataset)

        # Respublica
        else:
            project_path = '/home/guerramarj/github/{0}'.format(project_name, dataset)
            data_path = '/home/guerramarj/github/{0}/dataset/{1}'.format(project_name, dataset)

        # if no dataset location was given, then there will be extra symbols
        if not dataset:
            data_path = data_path[:-1]

    else:
        path = specific_path

    return project_path, data_path
