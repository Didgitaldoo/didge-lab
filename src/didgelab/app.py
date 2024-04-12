import logging
import sys
import os
from datetime import datetime
import configargparse
from multiprocessing import Manager

app = None

def init_app(name=None, create_output_folder=True):
    global app
    app = App(name=name, create_output_folder=create_output_folder)

def get_app():
    if app is None:
        init_app(None, True)
    return app

def get_config():
    return get_app().get_config()

class App:

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
            
            conf = self.get_config()
            conf_str = "Configuration:"
            for key in sorted(conf.keys()):
                conf_str += f"\n{key}: {conf[key]}"
            logging.info(conf_str)

        if "ipykernel" in sys.modules:
            self.start_message()

    # register services in the app to make them available from other parts of the application
    def register_service(self, service):
        self.services[type(service)] = service

    def get_service(self, service_type):
        if service_type in self.services.keys():
            return self.services[service_type]
        else:
            return None

    def get_config(self, path="config.ini"):
        
        if self.config is None:
            p = configargparse.ArgParser(default_config_files=['./*.conf'])
            
            p.add('-log_level', type=str, choices=["info", "error", "debug", "warn"], default="info", help='log level ')

            options = p.parse_known_args()[0]
            self.config = {}

            for key, value in vars(options).items():
                self.config[key]=value

        return self.config

    # publish / subscribe pattern for data exchange between services
    def publish(self, topic, args=None):
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
        if topic not in self.subscribers:
            self.subscribers[topic]=[]
        self.subscribers[topic].append(fct)


    def init_logging(self, filename="./log.txt", log_to_file=True):
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