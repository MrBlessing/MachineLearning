import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooh",fc=0.8)

#参数意义：注释名称，注释文本坐标，注释点坐标，
def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords="axes fraction",\
    xytext=centerPt,textcoords="axes fraction",va="center",ha="center",bbox=nodeType)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot
    fig.clf()
    #frameon表示是否覆盖其他图层
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode("决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode("叶节点",(0.5,0.1),(0.1,0.5),leafNode)