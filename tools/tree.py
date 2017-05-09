# !/bin/python3
####################### Example ###############################################
#def rootMethod(b):
#    return b

#def lower10(val):
#    if val < 10:
#        return (True, val)
#    return (False, val)

#def greater10(val):
#    if val > 10:
#        return (True, val)
#    return (False, val)

#def equal10(val):
#    if val == 10:
#        return (True, val)
#    return (False, val)

#def node1M(val):
#    if val >= 10:
#        return (True, val)
#    return (False, val)

#leaf1 = TreeLeaf(name="Lower", evalMethod=lower10)
#leaf2 = TreeLeaf(name="Greater", evalMethod=greater10)
#leaf3 = TreeLeaf(name="Equal", evalMethod=equal10)
#node1 = TreeNode(name='node1', children=[leaf2, leaf3], evalMethod=node1M)
#root = TreeNode(name='root', children=[leaf1, node1], evalMethod=rootMethod)
#print(root)
###############################################################################

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
            if self.enableLog:
                self.log.append(log)
        return passedEval

    def __str__(self):
        if self.enableLog:
            return '{}: {}   ({})'.format(self.name, self.result, self.log)
        return '{}: {}'.format(self.name, len(self.result))

