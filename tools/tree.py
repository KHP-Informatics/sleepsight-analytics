# !/bin/python3

class TreeNode:

    def __init__(self, name, children, evalMethod):
        self.name = name
        self.children = children
        self.evalMethod = evalMethod

    def invoke(self, parameters, log='/'):
        passedEval = self.evalMethod(parameters)
        if passedEval:
            for child in self.children:
                reachedLeaf = child.invoke(parameters, log='{}{}/'.format(log, self.name))
                if reachedLeaf:
                    break
        return passedEval

    def retrieveLeaves(self):
        leaves = []
        for child in self.children:
            tmpLeaf = TreeLeaf('mock', [])
            if type(child) is type(tmpLeaf):
                leaves.append(child)
            else:
                deeperLeaves = child.retrieveLeaves()
                leaves = leaves + deeperLeaves
        return leaves

    def __str__(self):
        leaves = self.retrieveLeaves()
        rendered = "\n{}-node Summary:\n".format(str.capitalize(self.name))
        for i in range(0, len(leaves)):
            rendered += '{} | {}\n'.format(i, leaves[i].__str__())
        rendered += '\n'
        return rendered



class TreeLeaf:

    def __init__(self, name, evalMethod, enableLog=False):
        self.name = name
        self.evalMethod = evalMethod
        self.enableLog = enableLog
        self.log = []
        self.result = []

    def invoke(self, parameters, log='/'):
        passedEval, result = self.evalMethod(parameters)
        if passedEval:
            self.result.append(result)
            self.log.append(log)
        return passedEval

    def __str__(self):
        if self.enableLog:
            return '{}: {}   ({})'.format(self.name, self.result, self.log)
        return '{}: {}'.format(self.name, self.result)

