"""Centralized Qt style sheet for the operator interface."""

APP_STYLE = """
QMainWindow, QWidget#root {
    background: #F2F5F8;
    color: #17202A;
    font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
    font-size: 14px;
}
QFrame#header, QFrame#panel, QFrame#videoPanel {
    background: #FFFFFF;
    border: 1px solid #DCE3EA;
    border-radius: 10px;
}
QLabel#title {
    color: #14213D;
    font-size: 22px;
    font-weight: 700;
}
QLabel#subtitle, QLabel#muted {
    color: #687786;
}
QLabel#sectionTitle {
    color: #243447;
    font-size: 15px;
    font-weight: 700;
}
QLabel#detectionCount {
    color: #176B45;
    background: #E3F5EA;
    border: 1px solid #B9E5CA;
    border-radius: 10px;
    padding: 3px 9px;
    font-weight: 650;
}
QLabel#video {
    background: #101820;
    color: #AAB7C4;
    border: 1px solid #253442;
    border-radius: 8px;
}
QLabel#statusPill {
    border-radius: 12px;
    padding: 4px 11px;
    font-weight: 600;
}
QLabel#statusPill[level="idle"] {
    color: #52616F;
    background: #E9EEF3;
}
QLabel#statusPill[level="ready"] {
    color: #176B45;
    background: #DCF5E8;
}
QLabel#statusPill[level="running"] {
    color: #1559A0;
    background: #DCEBFF;
}
QLabel#statusPill[level="error"] {
    color: #A42A2A;
    background: #FCE2E2;
}
QPushButton {
    min-height: 42px;
    padding: 0 16px;
    border-radius: 7px;
    font-weight: 650;
}
QPushButton#startButton {
    color: #FFFFFF;
    background: #178552;
    border: 1px solid #126D43;
}
QPushButton#startButton:hover { background: #126F45; }
QPushButton#startButton:pressed { background: #0F5E3A; }
QPushButton#stopButton {
    color: #FFFFFF;
    background: #C23B3B;
    border: 1px solid #A92E2E;
}
QPushButton#stopButton:hover { background: #A93030; }
QPushButton#secondaryButton {
    color: #33495E;
    background: #FFFFFF;
    border: 1px solid #BFCAD4;
}
QPushButton#secondaryButton:hover { background: #F3F6F8; }
QPushButton:disabled {
    color: #94A0AA;
    background: #E5E9ED;
    border-color: #D5DBE0;
}
QPlainTextEdit {
    color: #DCE6EE;
    background: #17212B;
    border: 1px solid #2A3A49;
    border-radius: 7px;
    padding: 7px;
    font-family: "Cascadia Mono", "Noto Sans Mono CJK SC", monospace;
    font-size: 12px;
}
QSplitter::handle { background: transparent; width: 8px; }
"""
