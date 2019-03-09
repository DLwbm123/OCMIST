from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from MainWindow import Ui_MainWindow

import os
import nibabel as nib
import numpy as np

VIEWER_SIZE = 384
IMAGE_SIZE = 128
MINI_VIEWER_SIZE = 160


class Image():
    def __init__(self, nii_image, name):
        self.name = name
        self.dim = nii_image.header.get_data_shape()
        self.dtype = nii_image.header.get_data_dtype()
        self.spacing = nii_image.header.get_zooms()
        self.data = np.array(nii_image.get_data())
        self.max = np.max(self.data)
        self.min = np.min(self.data)
        if self.name in ['flair', 't1', 't1ce', 't2', 'truth', 'pred']:
            self.toCharData()
        print(self.name, self.dim, self.dtype, self.spacing, self.max, self.min)

    def toCharData(self):
        self.char_data = self.data.copy()
        self.char_data -= self.min
        if self.name in ['flair', 't1', 't1ce', 't2']:
            self.char_data *= 255 / (self.max - self.min)
        self.char_data = self.char_data.astype(np.uint8)
        # print(self.char_data)

class ImageManager():
    def __init__(self, main_window):
        self.path = ''
        self.main_window = main_window

    def openProject(self, path):
        self.path = path
        print(path)
        # 检查该文件夹下的data_{flair, t1, t1ce, t2, truth}.nii.gz
        self.image_path = {
            'flair': os.path.join(self.path, 'data_flair.nii.gz'),
            't1': os.path.join(self.path, 'data_t1.nii.gz'),
            't1ce': os.path.join(self.path, 'data_t1ce.nii.gz'),
            't2': os.path.join(self.path, 'data_t2.nii.gz'),
            'truth': os.path.join(self.path, 'truth.nii.gz'),
            'pred': os.path.join(self.path, 'prediction.nii.gz')
        }
        # 不符合条件则返回并提示
        self.image_file = dict()
        self.image = dict()
        for f in self.image_path:
            if os.path.isfile(self.image_path[f]):
                self.image_file[f] = nib.load(self.image_path[f])
                self.image[f] = Image(self.image_file[f], f)
            else:
                QMessageBox.information(self.main_window, 'Information', f + ' file does not exist.', QMessageBox.Ok)
                return
        # 读取现有mark文件并写入到main_window.mark里
        self.main_window.mark = [list() for x in range(IMAGE_SIZE)]

        # 检查如果有up{1, 2, 4}.nii则加载，没有则创建空的
        # self.edit_file = dict()
        # self.edit = dict()
        # self.edit_path = {
        #     'up1': os.path.join(self.path, 'up1.nii'),
        #     'up2': os.path.join(self.path, 'up2.nii'),
        #     'up4': os.path.join(self.path, 'up4.nii')
        # }
        # for f in self.edit_path:
        #     if os.path.isfile(self.edit_path[f]):
        #         self.edit_file[f] = nib.load(self.edit_path[f])
        #         self.edit[f] = Image(self.edit_file[f], f)
        #     else:
        #         pass
        # 然后显示中间的切片，允许响应用户操作
        self.main_window.statusBar.showMessage('Project loaded: ' + path)
        self.main_window.loadFinished()

class Viewer(QLabel):
    mousemove = pyqtSignal(QEvent)
    mouseleave = pyqtSignal()
    mousepress = pyqtSignal(QEvent)

    def __init__(self, parent):
        super(Viewer, self).__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        self.mousemove.emit(event)
        # if event.buttons() == Qt.NoButton:
        #     pass
        # else:
        #     print(event.x(), event.y())

    def mousePressEvent(self, event):
        self.mousepress.emit(event)

    def leaveEvent(self, event):
        self.mouseleave.emit()

class Mask():
    # 在每个viewer上面都有一个相同尺寸的mask，可以在其上绘制各种图形
    def __init__(self, parent):
        self.mask = QLabel(parent)
        self.mask.setMouseTracking(True)
        self.mask.setGeometry(parent.geometry())
        self.mask.setAttribute(Qt.WA_TranslucentBackground, True)
        self.w = self.mask.width()
        self.h = self.mask.height()
        self.cross = [-100, -100]
        self.paint()

    def paint(self):
        self.img = np.zeros([self.w, self.h, 4], dtype=np.uint8)

        # draw code here
        self.drawCross(self.cross, [128, 255, 128, 128])

        qimg = QImage(self.img, self.w, self.h, self.w * 4, QImage.Format_RGBA8888)
        qpix = QPixmap(qimg)
        self.mask.setPixmap(qpix)

    def drawCross(self, pos, color):
        x = pos[0]
        y = pos[1]
        for j in range(y - 1, y + 2):
            for i in range(x - 8, x - 3):
                self.drawPixel(i, j, color)
            for i in range(x + 4, x + 9):
                self.drawPixel(i, j, color)
        for i in range(x - 1, x + 2):
            for j in range(y - 8, y - 3):
                self.drawPixel(i, j, color)
            for j in range(y + 4, y + 9):
                self.drawPixel(i, j, color)


    def drawPixel(self, x, y, color):
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return
        self.img[x, y] = color

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.container_score.setVisible(False)
        self.container_cor.setVisible(False)
        self.container_sag.setVisible(False)

        self.image_loaded = False
        self.image_manager = ImageManager(self)
        self.show_gt = False
        self.show_pred = True

        ### Setup operations.
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.button_open.pressed.connect(self.actionOpenProject)
        self.button_pred.pressed.connect(self.switchShowPred)
        self.button_gt.pressed.connect(self.switchShowGt)
        for v in [self.viewer_t1, self.viewer_t1ce, self.viewer_t2, self.viewer_flair]:
            v.mousemove.connect(self.viewerMouseMove)
            v.mouseleave.connect(self.viewerMouseLeave)
            v.mousepress.connect(self.viewerMousePress)

        ### Setup actions
        self.action_open.triggered.connect(self.actionOpenProject)

        self.addOverlays()
        self.show()

    def wheelEvent(self, event):
        if not self.image_loaded:
            return
        v = self.slider.value()
        if event.angleDelta().y() > 0:
            v += 1
        elif event.angleDelta().y() < 0:
            v -= 1
        v = min(v, self.slider.maximum())
        v = max(v, self.slider.minimum())
        self.slider.setValue(v)
        self.sliderValueChanged(v)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_O:
            path = 'Brats18_CBICA_ABB_1'
            self.image_manager.openProject(path)
        if event.key() == Qt.Key_Escape:
            self.close()

        if not self.image_loaded:
            return

        if event.key() == Qt.Key_Space:
            self.show_pred ^= True
            self.plotAll()
        if event.key() == Qt.Key_Tab:
            self.show_gt ^= True
            self.plotAll()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        if not self.image_loaded:
            return
        if event.key() == Qt.Key_Space:
            self.show_pred ^= True
            self.plotAll()
        if event.key() == Qt.Key_Tab:
            self.show_gt ^= True
            self.plotAll()

    def viewerMouseMove(self, event):
        if not self.image_loaded:
            return
        for m in self.mask:
            self.mask[m].cross = [event.y(), event.x()]
            self.mask[m].paint()

    def viewerMouseLeave(self):
        if not self.image_loaded:
            return
        for m in self.mask:
            self.mask[m].cross = [-100, -100]
            self.mask[m].paint()

    def viewerMousePress(self, event):
        if not self.image_loaded:
            return
        if event.button() == Qt.LeftButton:
            m = '+'
        elif event.button() == Qt.RightButton:
            m = '-'
        else:
            return
        self.addTemporaryMark(event.y(), event.x(), m)

    def addTemporaryMark(self, screeny, screenx, mark):
        # 首先转换为以viewer中心为原点
        screeny -= VIEWER_SIZE / 2
        screenx -= VIEWER_SIZE / 2
        # 然后转换为图像的坐标
        image = self.image_manager.image['t1']
        scale = image.spacing[1] / image.spacing[0]
        w = VIEWER_SIZE / max(scale, 1)
        h = VIEWER_SIZE * min(scale, 1)
        imagey = int(round((screeny / h + 0.5) * IMAGE_SIZE))
        imagex = int(round((screenx / w + 0.5) * IMAGE_SIZE))
        if imagex < 0 or imagey < 0 or imagex >= IMAGE_SIZE or imagey >= IMAGE_SIZE:
            return
        imagez = self.slice
        self.mark[imagez].append({
            'x': imagex,
            'y': imagey,
            'z': imagez,
            'm': mark
        })
        self.plotAll()
        print(imagex, imagey, imagez, mark)

    def addOverlays(self):
        self.mask = dict()
        self.mask['t1'] = Mask(self.viewer_t1)
        self.mask['flair'] = Mask(self.viewer_flair)
        self.mask['t1ce'] = Mask(self.viewer_t1ce)
        self.mask['t2'] = Mask(self.viewer_t2)

    def actionOpenProject(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.image_manager.openProject(path)

    def switchShowGt(self):
        self.show_gt ^= True
        self.plotAll()

    def switchShowPred(self):
        self.show_pred ^= True
        self.plotAll()

    def getViewer(self, name):
        if name == 'flair':
            return self.viewer_flair
        if name == 't1':
            return self.viewer_t1
        if name == 't1ce':
            return self.viewer_t1ce
        if name == 't2':
            return self.viewer_t2

    def plotImage(self, image, slice, direction = 'axi', size = VIEWER_SIZE):
        if direction == 'axi':
            qlabel = self.getViewer(image.name)
            img_gray = np.rot90(image.char_data[:, :, slice], -1).copy()
        elif direction == 'sag':
            qlabel = self.label_sag
            img_gray = np.flip(np.rot90(image.char_data[:, slice, :]), axis = 1).copy()
        elif direction == 'cor':
            qlabel = self.label_cor
            img_gray = np.flip(np.rot90(image.char_data[slice, :, :]), axis = 1).copy()

        img = np.stack((img_gray,) * 3, axis = -1)
        if direction == 'axi' and self.show_gt:
            gt = np.rot90(self.image_manager.image['truth'].char_data[:, :, slice], -1).copy()
            gt *= 255
            gt = np.max([gt, img_gray], axis = 0)
            img[:, :, 0] = gt
        if direction == 'axi' and self.show_pred:
            pred = np.rot90(self.image_manager.image['pred'].char_data[:, :, slice], -1).copy()
            pred *= 255
            pred = np.max([pred, img_gray], axis = 0)
            img[:, :, 2] = pred
        if direction == 'axi':
            for m in self.mark[slice]:
                self.drawMark(img, m['y'], m['x'], m['m'])
        if direction != 'axi':
            img[-1 - self.slice, :, 1] = 255

        qimg = QImage(img, image.dim[0], image.dim[1], QImage.Format_RGB888)
        qpix = QPixmap(qimg)
        if direction == 'axi':
            scale = image.spacing[1] / image.spacing[0]
        if direction == 'sag':
            scale = image.spacing[2] / image.spacing[0]
        if direction == 'cor':
            scale = image.spacing[2] / image.spacing[1]

        qpix = qpix.scaled(QSize(size / max(scale, 1), size * min(scale, 1)), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        qlabel.setPixmap(qpix)

    def plotSagCor(self):
        # 在界面左侧显示另外两个方向的切片
        self.plotImage(self.image_manager.image['t1'], 64, direction = 'sag', size = MINI_VIEWER_SIZE)
        self.plotImage(self.image_manager.image['t1'], 64, direction = 'cor', size = MINI_VIEWER_SIZE)

    def plotAll(self):
        self.plotImage(self.image_manager.image['flair'], self.slice)
        self.plotImage(self.image_manager.image['t1'], self.slice)
        self.plotImage(self.image_manager.image['t1ce'], self.slice)
        self.plotImage(self.image_manager.image['t2'], self.slice)
        self.plotSagCor()

    def drawMark(self, img, y, x, m):
        for i in range(x - 2, x + 3):
            if i >= 0 and i < IMAGE_SIZE:
                img[y][i] = [0, 255, 0]
        if m == '+':
            for i in range(y - 2, y + 3):
                if i >= 0 and i < IMAGE_SIZE:
                    img[i][x] = [0, 255, 0]


    def sliderValueChanged(self, value):
        # print(value)
        if not self.image_loaded:
            return
        self.slice = value - 1
        self.plotAll()
        self.label_slice.setText('Current slice: ' + str(value))

    def calculateDiceJaccardPrecisionRecall(self):
        gt = self.image_manager.image['truth'].char_data.astype(np.float)
        pred = self.image_manager.image['pred'].char_data.astype(np.float)
        TP = np.sum(gt * pred)
        FN = np.sum((gt - pred > 0).astype(np.float))
        FP = np.sum((gt - pred < 0).astype(np.float))
        SGT = np.sum(gt)
        SPRED = np.sum(pred)
        dice = 2 * TP / (SGT + SPRED)
        jaccard = TP / (SGT + SPRED - TP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        self.label_dice.setText('Dice: ' + str(round(dice, 3)))
        self.label_jaccard.setText('Jaccard: ' + str(round(jaccard, 3)))
        self.label_precision.setText('Precision: ' + str(round(precision, 3)))
        self.label_recall.setText('Recall: ' + str(round(recall, 3)))
        TP = round(TP, 2) * 100
        FN = round(FN, 2) * 100
        FP = round(FP, 2) * 100
        s = [FP, TP, TP, FN]
        for i in range(4):
            self.bar_dice.setStretch(i, s[i])
        s = [FP, TP, FN]
        for i in range(3):
            self.bar_jaccard.setStretch(i, s[i])
        s = [FP, TP]
        for i in range(2):
            self.bar_precision.setStretch(i, s[i])
        s = [TP, FN]
        for i in range(2):
            self.bar_recall.setStretch(i, s[i])

    def loadFinished(self):
        self.image_loaded = True
        self.sliderValueChanged(self.slider.value())
        self.calculateDiceJaccardPrecisionRecall()
        self.container_score.setVisible(True)
        self.container_sag.setVisible(True)
        self.container_cor.setVisible(True)



if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("")
    window = MainWindow()
    app.exec_()
