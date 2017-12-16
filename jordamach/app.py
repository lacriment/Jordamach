import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import numpy as np
import matplotlib.pyplot as plt

from ui.design import UiMainWindow
from linear_regression import Regrezio


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        # Set up the user interface from Designer.
        self.ui = UiMainWindow()
        self.ui.setupUi(self)

        self.r = Regrezio()

        self.ui.tableWidget.setColumnCount(6)
        self.ui.tableWidget.setHorizontalHeaderLabels(['Y', 'X1', 'X2', 'X3', 'X4', 'X5'])
        self.ui.tableWidget.setRowCount(50)

        # Connect up the buttons.
        self.ui.btn_fitModel.clicked.connect(self.fit_model)
        self.ui.btn_plot.clicked.connect(self.plot_model)
        self.ui.btn_clear.clicked.connect(self.clear_table)
        self.ui.btn_predict.clicked.connect(self.predict_value)
        self.ui.btn_exit.clicked.connect(QtCore.QCoreApplication.instance().quit)

    def fit_model(self):
        row_count = self.ui.tableWidget.rowCount()
        y = []
        x = []
        for row in range(0, row_count):
            y.append(self.ui.tableWidget.item(row, 0))
            x.append(self.ui.tableWidget.item(row, 1))
        y = np.array([[float(i.text())] for i in y if i is not None or i == ""])
        x = np.array([[float(i.text())] for i in x if i is not None or i == ""])

        self.r.fit(x, y)
        self.r.predict(self.r.x)
        self.ui.lbl_model.\
            setText("ŷ = %s + %s * X1" %
                    (round(float(self.r.intercept), 4), round(float(self.r.coefficients[0]), 4)))
        self.ui.lbl_rsqr.setText(str(self.r.score))
        print("x  :", self.r.x)
        print("y  :", self.r.y)
        print("y^ :", self.r.y_pred)

    def plot_model(self):
        plt.scatter(self.r.x, self.r.y, c="orange")
        plt.plot(self.r.x, self.r.y_pred, c="darkblue")
        plt.show()

    def predict_value(self):
        val = float(self.ui.txt_predict.text())
        predicted_val = self.r.y_func(val)
        self.ui.lbl_predict.setText(str(round(float(predicted_val), 4)))

    def clear_table(self):
        self.ui.tableWidget.clear()
        self.ui.tableWidget.setColumnCount(6)
        self.ui.tableWidget.setHorizontalHeaderLabels(['Y', 'X1', 'X2', 'X3', 'X4', 'X5'])
        self.ui.tableWidget.setRowCount(50)
        self.ui.lbl_model.clear()
        self.ui.lbl_predict.clear()
        self.ui.lbl_rsqr.clear()


def main():
    app = QApplication(sys.argv)
    form = App()
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
