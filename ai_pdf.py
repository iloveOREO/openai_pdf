# coding: utf-8

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QFileDialog
from ui_aipdf import Ui_AIPDF
import PyPDF2
from openai import OpenAI
import numpy as np
import re
import os
from PyQt6.QtCore import QThread

CHAT_MODEL = "gpt-4-0125-preview"
EMBEDDING_MODEL = "text-embedding-3-large"
api_key = 'xxxx'
local_key = os.path.join(os.path.dirname(__file__), 'aipdf.ini')

if os.path.exists(local_key):
    with open(local_key, 'r') as f:
        api_key = f.read()


class GPTAnswerSignals(QtCore.QObject):
    answer_generated = QtCore.pyqtSignal(str)


class GPTAnswer(QtCore.QRunnable):
    def __init__(self, question, closest_sentences, client):
        QThread.__init__(self)
        self.question = question
        self.closest_sentences = closest_sentences
        self.client = client
        self._is_running = False
        self.signals = GPTAnswerSignals()
        self.chat_model = CHAT_MODEL

    def run(self):
        self._is_running = True
        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": "\n".join(self.closest_sentences)}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                self.signals.answer_generated.emit(chunk.choices[0].delta.content)
                print(chunk.choices[0].delta.content, end="")
        self._is_running = False


class AIPDF(QtWidgets.QMainWindow, Ui_AIPDF):
    text_updated = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super(AIPDF, self).__init__(parent)
        self.setupUi(self)
        self.open_pdf_btn.clicked.connect(self.open_pdf)
        self.api_key = api_key
        self.openai_key_edit.setText(self.api_key)
        self.openai_key_edit.setEnabled(False)
        self.save_openai_btn.clicked.connect(self.change_openai_key)
        self.embedding_model = EMBEDDING_MODEL
        self.text_updated.connect(self.update_text)
        self.answer_browser.ensureCursorVisible()
        self.question_btn.clicked.connect(self.question_action)
        self.qthread_pool = QtCore.QThreadPool()
        self.gpt_answer_thread = GPTAnswer(None, None, None)
        self.gpt_answer_thread.signals.answer_generated.connect(self.update_stream_text)
        self.init_openai()

    def update_text(self, text):
        self.answer_browser.append(text)

    def update_stream_text(self, text):
        self.answer_browser.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        precursor = self.answer_browser.textCursor()
        pos = precursor.position()
        self.update_text(text)
        if pos == 0:
            return
        precursor.setPosition(pos)
        self.answer_browser.setTextCursor(precursor)
        self.answer_browser.textCursor().deleteChar()       
    
    def init_openai(self):
        api_key = self.openai_key_edit.text()
        if not api_key.startswith('sk-'):
            QtWidgets.QMessageBox.warning(self, "警告", "无效的sk, 请输入'sk-'开头的sk")
            self.client = None
            return
        with open(local_key, 'w') as f:
            f.write(api_key)
        self.client = OpenAI(api_key=api_key)
        self.gpt_answer_thread.client = self.client

    def change_openai_key(self, checked: bool):
        print(f'change_openai_key checked: {checked}')
        if checked:
            self.save_openai_btn.setText("保存")
            self.openai_key_edit.setEnabled(True)
        else:
            self.save_openai_btn.setText("设置OPENAI_KEY")
            self.openai_key_edit.setEnabled(False)
            self.init_openai()
    
    def get_chapters(self, file_path):
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        # 去掉引用部分
        references_index = text.find('References')
        if references_index != -1:
            text = text[:references_index]
        # 假设每个章节都以数字开头
        chapters = re.split(r'\n(?=\d+\.)', text)
        return chapters
    
    def refine_docs(self, documents):
        refined_docs = []
        for doc in documents:
            doc = doc.replace('\n', '')
            if refined_docs and len(refined_docs[-1]) < 100:
                refined_docs[-1] += doc
            else:
                refined_docs.append(doc)
        self.text_updated.emit(f'refined_docs: {refined_docs}, chapters num: {len(refined_docs)}')  
        return refined_docs

    def get_embedding(self, text):
        text = text if isinstance(text, list) else [text]
        return self.client.embeddings.create(input=text, model=self.embedding_model)

    def open_pdf(self):
        if self.client is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先设置正确的OPENAI sk")
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "PDF Files (*.pdf)")
        if not file_name:
            return
        self.pdf_path_label.setText(file_name)
        self.update()
        documents = self.get_chapters(file_name)
        self.refined_docs = self.refine_docs(documents)
        self.text_updated.emit("正在获取Embeddings...")
        response = self.get_embedding(self.refined_docs)
        self.embeddings = np.array([data.embedding for data in response.data])
        self.text_updated.emit(f'embeddings: {self.embeddings}')
        return self.refined_docs
    
    def question_action(self, checked: bool):
        if self.client is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先设置正确的OPENAI sk")
            return
        print(f'question_action checked: {checked}')
        question = self.question_input.text()
        if question.strip() == "":
            return
        if self.gpt_answer_thread._is_running:
            return
        
        # 当有一个新的问题时，将问题转换为嵌入
        response = self.get_embedding(question)
        question_embedding = np.array(response.data[0].embedding)
        # 找出与问题嵌入最接近的文档部分嵌入
        distances = np.linalg.norm(self.embeddings - question_embedding, axis=1)
        closest_sentence_indices = np.argsort(distances)[:3]
        closest_sentences = [self.refined_docs[i] for i in closest_sentence_indices]
        self.text_updated.emit(f'引用部分: 「{closest_sentences}」')
        self.text_updated.emit(f"正在回答问题: 「{question}」，请稍等...\n")
        self.gpt_answer_thread.question = question
        self.gpt_answer_thread.closest_sentences = closest_sentences
        self.qthread_pool.start(self.gpt_answer_thread)
    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = AIPDF()
    window.show()
    sys.exit(app.exec())
