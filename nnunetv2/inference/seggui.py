import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from predict import seg  # 导入 predict.py 的 main 函数


class SegmentationThread(QThread):
    finished_signal = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, input_folder, output_folder):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder

    def run(self):
        try:
            # 创建输出目录
            os.makedirs(self.output_folder, exist_ok=True)

            # 调用预测函数
            success, message = seg(self.input_folder, self.output_folder)
            self.finished_signal.emit(success, message)
        except Exception as e:
            self.finished_signal.emit(False, f"致命错误: {str(e)}")

class FolderSegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_folder = ""
        self.output_folder = ""

    def initUI(self):
        self.setWindowTitle("文件夹批量分割工具")
        self.setGeometry(500, 500, 1000, 200)

        layout = QVBoxLayout()

        # 显示文件夹路径
        self.folder_label = QLabel("请选择一个文件夹", self)
        self.folder_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.folder_label)

        # 选择文件夹按钮
        self.select_button = QPushButton("选择文件夹", self)
        self.select_button.clicked.connect(self.openFolderDialog)
        layout.addWidget(self.select_button)

        # 运行分割按钮
        self.process_button = QPushButton("开始处理", self)
        self.process_button.setEnabled(False)  # 初始禁用
        self.process_button.clicked.connect(self.run_segmentation)
        layout.addWidget(self.process_button)

        # 结果状态
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def openFolderDialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含图像的文件夹")
        if folder_path:
            self.selected_folder = folder_path
            self.output_folder = os.path.join(folder_path, "nnUNet_output")
            self.folder_label.setText(f"输入目录: {folder_path}\n输出目录: {self.output_folder}")
            self.process_button.setEnabled(True)

    def run_segmentation(self):
        if self.selected_folder:
            self.status_label.setText("处理中，请勿操作界面...")
            self.process_button.setEnabled(False)

            # 创建并启动线程
            self.thread = SegmentationThread(self.selected_folder, self.output_folder)
            self.thread.finished_signal.connect(self.on_seg_finished)
            self.thread.start()

    def on_seg_finished(self, success, message):
        self.process_button.setEnabled(True)
        self.status_label.setText(message)
        if success:
            # 自动打开输出文件夹（可选）
            if sys.platform == 'win32':
                os.startfile(self.output_folder)
            elif sys.platform == 'darwin':
                os.system(f'open "{self.output_folder}"')
            else:
                os.system(f'xdg-open "{self.output_folder}"')

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = FolderSegmentationApp()
#     window.show()
#     sys.exit(app.exec_())
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FolderSegmentationApp()
    window.show()
    sys.exit(app.exec_())
