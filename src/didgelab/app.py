"""
Application shell and configuration for DidgeLab.

Provides a singleton App with logging, output folders, config (e.g. from config.ini),
publish/subscribe for events (e.g. generation_ended), and service registration.
"""

import logging
import sys
import os
from datetime import datetime
import configargparse
from multiprocessing import Manager

app = None


def init_app(name=None, create_output_folder=True):
    """Create and set the global App. Call once at startup."""
    global app
    app = App(name=name, create_output_folder=create_output_folder)


def get_app():
    """Return the global App; initializes with default settings if not yet created."""
    if app is None:
        init_app(None, True)
    return app


def get_config():
    """Return the configuration dict from the global App."""
    return get_app().get_config()


class App:
    """
    Central app: logging, output directory, config, pub/sub, and service registry.
    """

    def __init__(self, name=None, create_output_folder=True):

        self.subscribers = {}
        self.output_folder = None
        self.services = {}
        self.config = None
        self.create_output_folder = create_output_folder
        
        if create_output_folder:
            if name is None:
                if "ipykernel" in sys.modules:
                    # we are calling from inside of a jupyter notebook
                    name = "jupyter"
                else:
                    # we are calling from python
                    name = os.path.basename(sys.argv[0])
                    if name.find(".")>0:
                        name = name[0:name.find(".")]
            outfolder=self.get_output_folder(suffix=name)
            log_file=os.path.join(outfolder, "log.txt")
            
            self.init_logging(filename=log_file, log_to_file=create_output_folder)

            self.start_message()
            logging.info(f"output folder: {outfolder}")
            
            conf = self.get_config()
            conf_str = "Configuration:"
            for key in sorted(conf.keys()):
                conf_str += f"\n{key}: {conf[key]}"
            logging.info(conf_str)

        if "ipykernel" in sys.modules:
            self.start_message()

    def register_service(self, service):
        """Register a service instance; retrieve later with get_service(type(service))."""
        self.services[type(service)] = service

    def get_service(self, service_type):
        if service_type in self.services.keys():
            return self.services[service_type]
        else:
            return None

    def get_config(self, path="config.ini"):
        """Load and cache config (e.g. from config.ini and ./ *.conf)."""
        
        if self.config is None:
            p = configargparse.ArgParser(default_config_files=['./*.conf'])
            
            p.add('-log_level', type=str, choices=["info", "error", "debug", "warn"], default="info", help='log level ')

            options = p.parse_known_args()[0]
            self.config = {}

            for key, value in vars(options).items():
                self.config[key]=value

        return self.config

    def publish(self, topic, args=None):
        """Notify all subscribers of topic (optional args passed to callbacks)."""
        logging.debug(f"self.publish topic={topic}, args={args}")
        if topic not in self.subscribers:
            return
        for s in self.subscribers[topic]:
            if args is None:
                s()
            elif type(args) == tuple:
                s(*args)
            else:
                s(args)

    def subscribe(self, topic, fct):
        """Register fct to be called when topic is published."""
        if topic not in self.subscribers:
            self.subscribers[topic]=[]
        self.subscribers[topic].append(fct)

    def init_logging(self, filename="./log.txt", log_to_file=True):
        """Configure root logger: file and console, level from config."""
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} %(message)s")
        rootLogger = logging.getLogger()

        if log_to_file:
            fileHandler = logging.FileHandler(filename)
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        level=self.get_config()["log_level"]
        if level == "info":
            rootLogger.setLevel(logging.INFO)
        elif level == "debug":
            rootLogger.setLevel(logging.DEBUG)
        elif level == "error":
            rootLogger.setLevel(logging.ERROR)
        elif level == "warn":
            rootLogger.setLevel(logging.WARN)

    def start_message(self):
        """Log ASCII banner and command line."""
        msg='''
 _____  _     _              _           _     
|  __ \(_)   | |            | |         | |    
| |  | |_  __| | __ _  ___  | |     __ _| |__  
| |  | | |/ _` |/ _` |/ _ \ | |    / _` | '_ \ 
| |__| | | (_| | (_| |  __/ | |___| (_| | |_) |
|_____/|_|\__,_|\__, |\___| |______\__,_|_.__/ 
                 __/ |                         
                |___/                          
'''
        msg += "Starting " + " ".join(sys.argv)
        logging.info(msg)

    def get_output_folder(self, suffix=""):
        """Create and return a timestamped output directory under evolutions/."""

        if self.output_folder is None:

            f = os.path.dirname(__file__)
            f = os.path.join(f, "../../evolutions/")

            if not os.path.exists(f):
                os.mkdir(f)

            my_date = datetime.now()

            folder_name=my_date.strftime('%Y-%m-%dT%H-%M-%S')
            if len(suffix)>0:
                folder_name += "_" + suffix

            config = self.get_config()
            if "log_folder_suffix" in config:
                folder_name += "_" + config["log_folder_suffix"]

            self.output_folder=os.path.join(f, folder_name)
            os.mkdir(self.output_folder)

        return self.output_folder