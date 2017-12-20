import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

from ui.design import Ui_MainWindow
from linear_regression import Regrezio


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.model_type = 'lin-lin'
        self.r = Regrezio(model=self.model_type)
        self.modelled = False
        self.msg = QMessageBox()

        self.ui.tableWidget.setColumnCount(6)
        self.ui.tableWidget.setHorizontalHeaderLabels(['Y', 'X1', 'X2', 'X3', 'X4', 'X5'])
        self.ui.tableWidget.setRowCount(50)

        # Connect up the buttons.
        self.ui.btn_fitModel.clicked.connect(self.fit_model)
        self.ui.btn_plot.clicked.connect(self.plot_model)
        self.ui.btn_clear.clicked.connect(self.clear_table)
        self.ui.btn_predict.clicked.connect(self.predict_value)
        self.ui.btn_exit.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.ui.cb_funcType.currentIndexChanged.connect(self.set_model_type)

    def set_model_type(self):
        self.model_type = self.ui.cb_funcType.currentText().lower()
        print(model_type)
        self.r = Regrezio(model=self.model_type)

    def keyPressEvent(self, event):
        row = self.ui.tableWidget.currentRow()
        col = self.ui.tableWidget.currentColumn()
        if event.key() == QtCore.Qt.Key_Backspace or event.key() == QtCore.Qt.Key_Delete:
            self.ui.tableWidget.setItem(row, col, QTableWidgetItem(""))

    def fit_model(self):
        row_count = 50
        y = []
        x = []
        for row in range(0, row_count):
            a = self.ui.tableWidget.item(row, 0)
            if a is not None:
                try:
                    if a.text() != "":
                        y.append(float(a.text()))
                except ValueError:
                    self.ui.statusBar.showMessage("Please enter numerical data into the table!")
            b = self.ui.tableWidget.item(row, 1)
            if b is not None:
                try:
                    if b.text() != "":
                        x.append(float(b.text()))
                except ValueError:
                    self.ui.statusBar.showMessage("Please enter numerical data into the table!")

        y = np.array([[i] for i in y])
        x = np.array([[i] for i in x])

        if len(x) > 1 and len(y) > 1 and len(y) == len(x):
            self.r.fit(x, y)
            self.r.predict(self.r.x)
            self.ui.lbl_model.\
                setText("ŷ = %s + %s * X1" %
                        (round(float(self.r.intercept), 4), round(float(self.r.coefficients[0]), 4)))
            self.ui.lbl_rsqr.setText(str(round(self.r.score, 4))) 
            self.modelled = True
            self.ui.statusBar.showMessage('Successfully accomplished.')
        else:
            self.modelled = False
            self.ui.statusBar.showMessage("Every column should have same number of data!")

    def plot_model(self):
        if self.modelled:
            y = np.array([i[0] for i in self.r.y])
            x = np.array([i[0] for i in self.r.x])
            ax = sns.regplot(x=x, y=y, color="g")
            plt.show()
        else:
            self.ui.statusBar.showMessage("Model must have been fit to make a plot.")

    def predict_value(self):
        if self.modelled:
            val = float(self.ui.txt_predict.text())
            predicted_val = self.r.y_func(val)
            self.ui.lbl_predict.setText(str(round(float(predicted_val), 4)))
        else:
            self.ui.statusBar.showMessage("Model must have been fit to make a prediction.")

    def clear_table(self):
        self.ui.tableWidget.clear()
        self.ui.tableWidget.setColumnCount(6)
        self.ui.tableWidget.setHorizontalHeaderLabels(['Y', 'X1', 'X2', 'X3', 'X4', 'X5'])
        self.ui.tableWidget.setRowCount(50)
        self.ui.lbl_model.setText("ŷ = ")
        self.ui.lbl_predict.clear()
        self.ui.lbl_rsqr.clear()


def main():
    app = QApplication(sys.argv)
    form = App()
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
