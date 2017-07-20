from datetime import datetime


class Logger:

    def __init__(self, path, filename, printLog=False):
        self.filename = filename
        self.dir = path
        self.path = self.dir + self.filename
        self.printLog = printLog
        self.checkFileExists()

    def checkFileExists(self):
        try:
            f = open(self.path, 'r')
            f.close()
        except FileNotFoundError:
            now = datetime.now()
            f = open(self.path, 'w')
            f.write('+++ FILE CREATED ON {} +++\n'.format(now))
            f.close()

    def emit(self, msg, newRun=False, indents=0):
        now = datetime.now()
        rendered = self.renderLog(now, msg, newRun, indents)
        if self.printLog:
            print(rendered)
        f = open(self.path, 'a')
        f.write(rendered)
        f.close()

    def renderLog(self, dateTime, msg, newRun=False, indents=0):
        renderedNewRun = '\n\n=== SLEEPSIGHT {} NEW RUN ===\n'.format(self.filename)
        indentations = self.renderIndentations(indents)
        rendered = ''
        if(newRun):
            rendered += renderedNewRun
        rendered += '{}\t{}{}\n'.format(dateTime, indentations, msg)
        return rendered

    def renderIndentations(self, indents):
        indent = '\t'
        rendered = ''
        for i in range(0, indents):
            rendered += indent
        return rendered

    def getLastMessage(self):
        f = open(self.path, 'r')
        log = f.read()
        f.close()
        logs = log.split('\n')
        return logs[len(logs)-2]