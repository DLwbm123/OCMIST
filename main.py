from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from MainWindow import Ui_MainWindow
from prediction import predict_one, prediction_to_image, load_trained_model

import os
import nibabel as nib
import numpy as np
import time
import math
import json
import csv
from random import randint

VIEWER_SIZE = 384
IMAGE_SIZE = 128
MINI_VIEWER_SIZE = 160

MODEL = 'pen'

TEST = False

class Image():
    def __init__(self, nii_image, name):
        self.name = name
        self.dim = nii_image.header.get_data_shape()
        self.dtype = nii_image.header.get_data_dtype()
        self.spacing = nii_image.header.get_zooms()
        self.data = np.array(nii_image.get_data())
        self.affine = nii_image.affine
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
        self.model = load_trained_model(MODEL)

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
            # 'pred': os.path.join(self.path, 'prediction.nii.gz')
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
        self.main_window.resize_level = 0
        self.main_window.stroke = [list() for x in range(IMAGE_SIZE)]

        self.main_window.operations = list()
        self.main_window.loadOperations(self.path, load_prediction = True)

        # 然后显示中间的切片，允许响应用户操作
        self.main_window.statusBar.showMessage('Project loaded: ' + path)
        self.main_window.loadFinished()

    def savePrediction(self):
        t = time.strftime("%Y%m%d%H%M%S", time.localtime())
        filename = 'prediction_' + t + '.nii.gz'
        path = os.path.join(self.path, filename)
        self.image_file['pred'].to_filename(path)
        self.main_window.saveOperations(filename)
        if not TEST:
            QMessageBox.information(self.main_window, 'Information', 'Prediction saved to ' + path, QMessageBox.Ok)

    def loadPrediction(self, filename):
        path = os.path.join(self.path, filename)
        self.image_path['pred'] = path
        if os.path.isfile(path):
            self.image_file['pred'] = nib.load(self.image_path['pred'])
            self.image['pred'] = Image(self.image_file['pred'], 'pred')

    def createEditImage(self, mark):
        self.edit = np.zeros([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]).astype(np.float)
        for n in mark:
            for m in n:
                x = IMAGE_SIZE - 1 - m['x'] # 空间需要翻转一下
                y = m['y']
                z = m['z']
                for i in range(x - 1, x + 2):
                    for j in range(y - 1, y + 2):
                        for k in range(z - 1, z + 2):
                            if i >= 0 and j >= 0 and k >= 0 and i < IMAGE_SIZE and j < IMAGE_SIZE and k < IMAGE_SIZE:
                                self.edit[i, j, k] = 1 if m['m'] == '+' else -1
        # if MODEL == 'up':
        self.edit *= 316.685 # magic

    def createAdjustImage(self, level):
        self.adjust = np.ones([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]).astype(np.float)
        self.adjust *= level
        # if MODEL == 'de':
        self.adjust -= 0.1298 # magic
        self.adjust /= 3 # magic

    # def createBoxImage(self, box):
    #     self.box = np.zeros([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]).astype(np.float)
    #     for b in box:
    #         A = [min(b['A'][0], b['B'][0]), min(b['A'][1], b['B'][1])]
    #         B = [max(b['A'][0], b['B'][0]), max(b['A'][1], b['B'][1])]
    #         z = b['z']
    #         m = b['m']
    #         for i in range(A[0], B[0] + 1):
    #             for j in range(A[1], B[1] + 1):
    #                 for k in range(z - 1, z + 2):
    #                     if i >= 0 and j >= 0 and k >= 0 and i < IMAGE_SIZE and j < IMAGE_SIZE and k < IMAGE_SIZE:
    #                         self.box[IMAGE_SIZE - 1 - i, j, k] = m
    #     self.box -= 0.00022
    #     self.box /= 0.01489611
    #     # self.box /= 3

    def createPenImage(self, stroke):
        self.pen = np.zeros([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]).astype(np.float)
        for n in stroke:
            for s in n:
                for p in s['p']:
                    x = IMAGE_SIZE - 1 - p['x'] # 空间需要翻转一下
                    y = p['y']
                    z = p['z']
                    for i in range(x - 1, x + 2):
                        for j in range(y - 1, y + 2):
                            for k in range(z - 1, z + 2):
                                if i >= 0 and j >= 0 and k >= 0 and i < IMAGE_SIZE and j < IMAGE_SIZE and k < IMAGE_SIZE:
                                    self.pen[i, j, k] = 1 if s['m'] == '+' else -1
        self.pen -= 0.000017638
        self.pen /= 0.00393432

    def predict(self):
        # prepare data:
        affine = self.image['t1'].affine
        data = [
            self.image['t1'].data,
            self.image['t1ce'].data,
            self.image['flair'].data,
            self.image['t2'].data,
        ]
        if MODEL == 'de' or MODEL == 'de_up' or MODEL == 'de_up_pen':
            data.append(self.adjust)
        if MODEL == 'up' or MODEL == 'de_up' or MODEL == 'de_up_pen':
            data.append(self.edit)
        if MODEL == 'pen' or MODEL == 'de_up_pen':
            data.append(self.pen)

        # 还需要加上用户交互数据
        prediction = predict_one(self.model, data, affine)
        self.image_file['pred'] = prediction_to_image(prediction, affine, labels = (1,), label_map = True)
        self.image['pred'] = Image(self.image_file['pred'], 'pred')
        self.main_window.plotAll()
        self.main_window.calculateDiceJaccardPrecisionRecall()




class Viewer(QLabel):
    mousemove = pyqtSignal(QEvent)
    mouseleave = pyqtSignal()
    mousepress = pyqtSignal(QEvent)
    mouserelease = pyqtSignal(QEvent)

    def __init__(self, parent):
        super(Viewer, self).__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        self.mousemove.emit(event)

    def mousePressEvent(self, event):
        self.mousepress.emit(event)

    def mouseReleaseEvent(self, event):
        self.mouserelease.emit(event)

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
        self.size = 0
        self.paint()

    def paint(self):
        self.img = np.zeros([self.w, self.h, 4], dtype=np.uint8)

        # draw code here
        self.drawCross(self.cross, [255, 254, 87, 128], self.size)
        # self.drawCross(self.cross, [170, 228, 184, 128], self.size)

        qimg = QImage(self.img, self.w, self.h, self.w * 4, QImage.Format_RGBA8888)
        qpix = QPixmap(qimg)
        self.mask.setPixmap(qpix)

    def drawCross(self, pos, color, size = 0):
        x = pos[0]
        y = pos[1]
        for j in range(y - 1, y + 2):
            for i in range(x - 8, x - 3):
                self.drawPixel(i - size * 4, j, color)
            for i in range(x + 4, x + 9):
                self.drawPixel(i + size * 4, j, color)
        for i in range(x - 1, x + 2):
            for j in range(y - 8, y - 3):
                self.drawPixel(i, j - size * 4, color)
            for j in range(y + 4, y + 9):
                self.drawPixel(i, j + size * 4, color)


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
        self.show_gt = True
        self.show_pred = True
        self.show_image = True

        self.mouse_strength = 0

        ### Setup operations.
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.button_open.pressed.connect(self.actionOpenProject)
        # self.button_pred.pressed.connect(self.switchShowPred)
        # self.button_gt.pressed.connect(self.switchShowGt)
        self.button_confirm.pressed.connect(self.apply)
        self.button_undo.pressed.connect(self.undo)
        self.button_save.pressed.connect(self.image_manager.savePrediction)
        for v in [self.viewer_t1, self.viewer_t1ce, self.viewer_t2, self.viewer_flair]:
            v.mousemove.connect(self.viewerMouseMove)
            v.mouseleave.connect(self.viewerMouseLeave)
            v.mousepress.connect(self.viewerMousePress)
            v.mouserelease.connect(self.viewerMouseRelease)

        ### Setup actions
        self.action_open.triggered.connect(self.actionOpenProject)
        self.action_save.triggered.connect(self.image_manager.savePrediction)
        self.action_pred.triggered.connect(self.switchShowPred)
        self.action_gt.triggered.connect(self.switchShowGt)
        self.action_image.triggered.connect(self.switchShowImage)
        self.action_undo.triggered.connect(self.undo)
        self.action_apply.triggered.connect(self.apply)
        self.action_extend.triggered.connect(self.extend)
        self.action_shrink.triggered.connect(self.shrink)

        # self.action_calculate_box.triggered.connect(self.calculateBox)

        self.addOverlays()
        self.show()

        if TEST:
            # self.runAllPrediction()
            self.runAllEvaluation()

    def wheelEvent(self, event):
        if not self.image_loaded:
            return
        if event.modifiers() & Qt.ControlModifier:
            # 微调分割边缘
            self.resize_level += 0.5 if event.angleDelta().y() > 0 else -0.5
            # self.resize_level = min(self.resize_level, 5)
            # self.resize_level = max(self.resize_level, -5)
            self.updateResizeLevel()
        else:
            # 切换切片
            v = self.slider.value()
            v += 1 if event.angleDelta().y() > 0 else -1
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
        if event.key() == Qt.Key_Up:
            self.mouse_strength = min(1, self.mouse_strength + 1)
        if event.key() == Qt.Key_Down:
            self.mouse_strength = max(-1, self.mouse_strength - 1)
        # if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
        #     self.apply()
        # if event.key() == Qt.Key_S:
        #     self.image_manager.savePrediction()
        # if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
        #     self.undo()


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
            self.mask[m].size = self.mouse_strength
            self.mask[m].paint()
        imagey, imagex = self.screenToImage(event.y(), event.x())
        self.updateMousePosition(imagey, imagex)
        if event.buttons() == Qt.LeftButton or event.buttons() == Qt.RightButton:
            self.addStrokePoint(imagey, imagex)
            self.plotAll()

    def viewerMouseLeave(self):
        if not self.image_loaded:
            return
        for m in self.mask:
            self.mask[m].cross = [-100, -100]
            self.mask[m].paint()
        self.updateMousePosition(-1, -1)

    def viewerMousePress(self, event):
        if not self.image_loaded:
            return
        if event.buttons() == Qt.LeftButton:
            self.addStroke('+')
        if event.buttons() == Qt.RightButton:
            self.addStroke('-')

    def viewerMouseRelease(self, event):
        if not self.image_loaded:
            return
        imagey, imagex = self.screenToImage(event.y(), event.x())
        self.addStrokePoint(imagey, imagex)
        self.checkStrokeLength()
        self.plotAll()

    def screenToImage(self, screeny, screenx):
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
            return -1, -1
        return imagey, imagex

    def addMark(self, imagey, imagex, mark, imagez = None, plot = False):
        # imagey, imagex = self.screenToImage(screeny, screenx)
        if imagey < 0 or imagex < 0:
            return
        if imagez == None:
            imagez = self.slice
        m = {
            'x': imagex,
            'y': imagey,
            'z': imagez,
            'm': mark
        }
        self.mark[imagez].append(m)
        self.operations.append({
            'operation': 'mark',
            'data': m
        })
        if plot:
            self.plotAll()
        self.updateNumberOfMarks()
        print('mark:', m)

    def addStroke(self, mark, imagez = None):
        if imagez == None:
            imagez = self.slice
        s = {
            'p': list(),
            'm': mark,
            'z': imagez
        }
        self.stroke[imagez].append(s)
        self.operations.append({
            'operation': 'stroke',
            'data': s
        })
        self.updateNumberOfStrokes()
        # print('addStroke:', mark)

    def addStrokePoint(self, imagey, imagex, imagez = None, plot = False):
        if imagey < 0 or imagex < 0:
            return
        if imagez == None:
            imagez = self.slice
        s = self.stroke[imagez][-1]
        interpolate = [[imagex, imagey]]
        # 对最后一个点和当前的点插值
        if len(s['p']) != 0:
            last = s['p'][-1]
            x1 = last['x']
            y1 = last['y']
            x2 = imagex
            y2 = imagey
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                for i in range(1, abs(dy)):
                    interpolate.append([x1, y1 + i * (1 if dy > 0 else -1)])
            elif dy == 0:
                for i in range(1, abs(dx)):
                    interpolate.append([x1 + i * (1 if dx > 0 else -1), y1])
            else:
                if dx < 0:
                    dx *= -1
                    dy *= -1
                    tmp = x1
                    x1 = x2
                    x2 = tmp
                    tmp = y1
                    y1 = y2
                    y2 = tmp
                k = dy / dx
                e = -0.5
                u = 1 if k > 0 else -1
                x = x1
                y = y1
                if abs(k) <= 1:
                    for i in range(dx + 1):
                        interpolate.append([x, y])
                        x += 1
                        e += u * k
                        if e >= 0:
                            y += u
                            e -= 1
                else:
                    for i in range(dy * u + 1):
                        interpolate.append([x, y])
                        y += u
                        e += 1 / k * u
                        if e >= 0:
                            x += 1
                            e -= 1

        s['p'].append({
            'x': imagex,
            'y': imagey,
            'z': imagez,
            'i': interpolate
        })
        if plot:
            self.plotAll()
        # print(imagex, imagey, self.slice, s['m'])

    def checkStrokeLength(self):
        # 检查最后生成的笔触，如果长度太短，就转换为标记
        length = 0
        s = self.stroke[self.slice][-1]
        for i in range(len(s['p']) - 1):
            length += abs(s['p'][i]['x'] - s['p'][i + 1]['x'])
            length += abs(s['p'][i]['y'] - s['p'][i + 1]['y'])
        if length <= 5:
            # print(s)
            x = s['p'][-1]['x']
            y = s['p'][-1]['y']
            m = s['m']
            self.undo()
            self.addMark(y, x, m, plot = True)
        else:
            print('stroke:', length)

    def shrink(self):
        self.resize_level -= 1.0

    def extend(self):
        self.resize_level += 1.0

    def apply(self):
        if not self.image_loaded:
            return
        # # 将当前self.tmp_mark里的数据全部转移至self.mark
        # for i in range(IMAGE_SIZE):
        #     self.mark[i] += self.tmp_mark[i]
        #     self.tmp_mark[i] = list()
        # 然后计算这一部分

        if MODEL == 'de' or MODEL == 'de_up' or MODEL == 'de_up_pen':
            self.image_manager.createAdjustImage(self.resize_level)
        if MODEL == 'up' or MODEL == 'de_up' or MODEL == 'de_up_pen':
            self.image_manager.createEditImage(self.mark)
        if MODEL == 'pen' or MODEL == 'de_up_pen':
            self.image_manager.createPenImage(self.stroke)
        self.image_manager.predict()

    def undo(self):
        if not self.image_loaded:
            return
        if len(self.operations) == 0:
            return
        o = self.operations[-1]
        if o['operation'] == 'mark':
            z = o['data']['z']
            self.mark[z].pop()
            self.plotAll()
            self.updateNumberOfMarks()
        if o['operation'] == 'adjust':
            pass
        if o['operation'] == 'stroke':
            z = o['data']['z']
            self.stroke[z].pop()
            self.plotAll()
            self.updateNumberOfStrokes()
        # print('undo:', self.operations[-1])
        self.operations.pop()

    def addOverlays(self):
        self.mask = dict()
        self.mask['t1'] = Mask(self.viewer_t1)
        self.mask['flair'] = Mask(self.viewer_flair)
        self.mask['t1ce'] = Mask(self.viewer_t1ce)
        self.mask['t2'] = Mask(self.viewer_t2)

    def actionOpenProject(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if len(path) == 0:
            return
        self.image_manager.openProject(path)

    def switchShowGt(self):
        if not self.image_loaded:
            return
        self.show_gt ^= True
        self.plotAll()

    def switchShowPred(self):
        if not self.image_loaded:
            return
        self.show_pred ^= True
        self.plotAll()

    def switchShowImage(self):
        if not self.image_loaded:
            return
        self.show_image ^= True
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
        if not self.show_image:
            img_gray *= 0
        img = np.stack((img_gray,) * 3, axis = -1)
        gt = np.rot90(self.image_manager.image['truth'].char_data[:, :, slice], -1).copy()
        pred = np.rot90(self.image_manager.image['pred'].char_data[:, :, slice], -1).copy()
        # gp = gt * pred
        if direction == 'axi' and self.show_gt:
            img[:, :, 0][gt == 1] = 255
            # self.mixColor(img, gt, [224, 37, 56])
        if direction == 'axi' and self.show_pred:
            img[:, :, 2][pred == 1] = 255
            # self.mixColor(img, pred, [48, 107, 200])
        if direction == 'axi' and self.show_pred and self.show_gt:
            pass
            # self.mixColor(img, gp, [255, 64, 255])#[224, 107, 200])

        if direction == 'axi':
            for m in self.mark[slice]:
                self.drawMark(img, m['y'], m['x'], m['m'], 'mark')
            for s in self.stroke[slice]:
                for m in s['p']:
                    for i in m['i']:
                        self.drawMark(img, i[1], i[0], s['m'], 'stroke')
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

    # def mixColor(self, img, mask, color):
    #     # img[mask == 1] = color
    #     # img[mask == 1] = (img[mask == 1] / 3).astype(np.uint8)
    #     # img[mask == 1] += (np.array(color) * 2 / 3).astype(np.uint8)

    def drawMark(self, img, y, x, m, type):
        color = [85, 248, 85] if m == '+' else [255, 226, 88]
        # color = [249, 167, 70] if m == '+' else [41, 200, 252]
        if type == 'mark':
            for i in range(x - 2, x + 3):
                if i >= 0 and i < IMAGE_SIZE:
                    img[y][i] = color
            if m == '+':
                for i in range(y - 2, y + 3):
                    if i >= 0 and i < IMAGE_SIZE:
                        img[i][x] = color
        # color = [249, 167, 70] if m == '+' else [41, 200, 252]
        # color = [253, 234, 114] if m == '+' else [87, 222, 253]
        if type == 'stroke':
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if i >= 0 and i < IMAGE_SIZE:
                        if j >= 0 and j < IMAGE_SIZE:
                            img[j, i] = color

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
        return [dice, jaccard, precision, recall]

    def updateMousePosition(self, y, x):
        if x < 0 or y < 0:
            self.label_xy.setVisible(False)
            return
        self.label_xy.setVisible(True)
        self.label_xy.setText('x: ' + str(x) + ', y: ' + str(y))

    def updateNumberOfStrokes(self):
        count = 0
        for i in self.stroke:
            count += len(i)
        self.label_stroke.setText('Number of strokes: ' + str(count))

    def updateNumberOfMarks(self):
        count = 0
        for i in self.mark:
            count += len(i)
        self.label_mark.setText('Number of marks: ' + str(count))

    def updateResizeLevel(self):
        self.label_resize.setText('Resize level: ' + str(self.resize_level))

    def loadFinished(self):
        self.image_loaded = True
        self.sliderValueChanged(self.slider.value())
        self.calculateDiceJaccardPrecisionRecall()
        self.container_score.setVisible(True)
        self.container_sag.setVisible(True)
        self.container_cor.setVisible(True)
        self.updateNumberOfMarks()
        self.updateNumberOfStrokes()
        self.updateResizeLevel()

    def saveOperations(self, prediction_filename):
        path = os.path.join(self.image_manager.path, MODEL + '.edit.json')
        if len(self.operations) == 0 and self.resize_level == 0:
            path = os.path.join(self.image_manager.path, MODEL + '.no_edit.json')
        data = json.dumps({
            'score': self.calculateDiceJaccardPrecisionRecall(),
            'prediction': prediction_filename,
            'resize': self.resize_level,
            'operations': self.operations
        })
        f = open(path, 'w')
        f.write(data)
        f.close()

    def loadOperations(self, folder, load_prediction = True):
        path = os.path.join(folder, MODEL + '.edit.json')
        if not os.path.isfile(path):
            path = os.path.join(folder, MODEL + '.no_edit.json')
        if os.path.isfile(path):
            f = open(path, 'r')
            data = json.loads(f.read())
            f.close()
            self.resize_level = data['resize']
            for o in data['operations']:
                m = o['data']['m']
                z = o['data']['z']
                if o['operation'] == 'stroke':
                    self.addStroke(m, imagez = z)
                    for p in o['data']['p']:
                        self.addStrokePoint(p['y'], p['x'], imagez = z)
                if o['operation'] == 'mark':
                    y = o['data']['y']
                    x = o['data']['x']
                    self.addMark(y, x, m, imagez = z)
            pred = data['prediction']
        else:
            pred = 'prediction.nii.gz'
        # self.plotAll()
        if load_prediction:
            self.image_manager.loadPrediction(pred)

    def runAllPrediction(self, path = '../../brats_naive124_half1'):
        folders = os.listdir(path)
        for f in folders:
            p = os.path.join(path, f)
            if not os.path.isfile(p):
                self.image_manager.openProject(p)
                self.apply()
                self.image_manager.savePrediction()

    def runAllEvaluation(self, path = '../../brats_naive124_half1'):
        folders = os.listdir(path)
        score = dict()
        for f in folders:
            p = os.path.join(path, f)
            if not os.path.isfile(p):
                score[f] = dict()
                for r in ['naive', 'de', 'up', 'pen', 'de_up_pen']:
                    file = open(os.path.join(path, f, r + '.no_edit.json'), 'r')
                    data = json.loads(file.read())
                    file.close()
                    score[f][r] = data['score']
                score[f]['subject'] = f
        for i, t in enumerate(['dice', 'jaccard', 'precision', 'recall']):
            out = open(os.path.join(path, t + '.csv'), 'w', newline='')
            writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['subject', 'naive', 'de', 'up', 'pen', 'de_up_pen'])
            for s in score:
                writer.writerow([(score[s][x] if x == 'subject' else score[s][x][i]) for x in ['subject', 'naive', 'de', 'up', 'pen', 'de_up_pen']])
            out.close()


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("")
    window = MainWindow()
    app.exec_()
