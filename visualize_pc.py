from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import argparse
from datetime import datetime

class plot3d(object):
    def __init__(self, title='null'):
        #
        self.glview = gl.GLViewWidget()
        coord = gl.GLAxisItem()
        coord.setSize(1,1,1)
        #self.glview.addItem(coord)
        self.glview.setMinimumSize(QtCore.QSize(600, 500))
        self.glview.pan(1, 0, 0)
        self.glview.setCameraPosition(azimuth=180)
        self.glview.setCameraPosition(elevation=0)
        self.glview.setCameraPosition(distance=5)
        self.items = []
        #
        self.view = QtGui.QWidget()
        self.view.window().setWindowTitle(title)
        hlayout = QtGui.QHBoxLayout()
        snap_btn = QtGui.QPushButton('&Snap')
        def take_snap():
            curr_time = datetime.now().strftime('%Y%m%d%H%M%S')
            save_file_str = curr_time + '.jpg'
            qimg = self.glview.readQImage()
            qimg.save(save_file_str)
        snap_btn.clicked.connect(take_snap)
        hlayout.addWidget(snap_btn)
        hlayout.addStretch()
        layout = QtGui.QVBoxLayout()
        #
        layout.addLayout(hlayout)
        layout.addWidget(self.glview)
        self.view.setLayout(layout)
    def add_item(self, item):
        self.glview.addItem(item)
        self.items.append(item)
    def clear(self):
        for it in self.items:
            self.glview.removeItem(it)
        self.items.clear()
    def add_points(self, points, colors):
        points_item = gl.GLScatterPlotItem(pos=points, size=1.5, color=colors)
        self.add_item(points_item)
    def add_line(self, p1, p2, color, width=3):
        lines = np.array([[p1[0], p1[1], p1[2]],
                          [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines',
                                       color=color, width=width, antialias=True)
        self.add_item(lines_item)
    def plot_bbox_mesh(self, gt_boxes3d, color=(0,1,0,1)):
        b = gt_boxes3d
        for k in range(0,4):
            i,j=k,(k+1)%4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)
            i,j=k+4,(k+1)%4 + 4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)
            i,j=k,k+4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)

def value_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = (pc_inte - minimum + 0.1) / (maximum - minimum + 0.1)
    r = (np.maximum((1 - ratio), 0))
    b = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b]).transpose()

def view_points_cloud(pc_velo=None, pc_pseudo=None):
    app = QtGui.QApplication([])
    glview = plot3d()

    pc_velo_color = np.ones([pc_velo.shape[0], 4])
    pc_velo_color[:,0] = 0.0 # blue
    glview.add_points(pc_velo, pc_velo_color)

    pc_pseudo_color = np.ones([pc_pseudo.shape[0],4])
    pc_pseudo_color[:,2] = 0.0 # yellow
    glview.add_points(pc_pseudo, pc_pseudo_color)
    glview.view.show()
    return app.exec()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--velo', type=str, default='~/Kitti/object/training/calib')
    parser.add_argument('--pseudo', type=str)
    args = parser.parse_args()

    v_pc = np.fromfile(args.velo, dtype=np.float32).reshape((-1, 4))[:, :3]
    p_pc = np.fromfile(args.pseudo, dtype=np.float32).reshape((-1, 4))[:, :3]

    print(f'v_pc shape = {v_pc.shape} and p_pc shape = {p_pc.shape}')

    # time to visualize
    view_points_cloud(v_pc, p_pc)