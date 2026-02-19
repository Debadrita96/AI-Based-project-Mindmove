import pickle
import socket
import time
from pathlib import Path

import numpy as np
import select


class RealTimeDataset:
    def __init__(self, data_path: Path, task: str):
        self.emg = np.concatenate(
            pickle.load(data_path.open("rb"))[task][:, 0], axis=-1
        )

        temp = np.zeros((408, self.emg.shape[1]), dtype=np.int16)
        temp[:64] = self.emg[:64]
        temp[128:384] = self.emg[64:]

        self.emg = temp


# Class for EMG simulator
class EMGSimulator:
    def __init__(self, tcp_ip="localhost", tcp_port=31000, fread=8, emg_data=None):
        # Socket settings
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.server_socket = None
        self.sockets_list = []
        self.read_buffer = 10

        # Streaming characteristics
        self.fread = fread
        self.sampling_frequency = 2048
        self.frame_len = self.sampling_frequency / self.fread
        self.frame_time = 1 / self.fread
        self.frame_counter = 0

        self.start_command = "startTX"
        self.stop_command = "stopTX"
        self.feedback_msg = "OTBiolab".encode("utf-8")
        self.client_connected = False

        # Data buffer
        self.emg_data = emg_data

    def open_connection(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.tcp_ip, self.tcp_port))
        self.server_socket.listen()
        self.server_socket.setblocking(True)

        self.sockets_list.append(self.server_socket)

    def close_connection(self):
        self.server_socket.close()

    def stream_control(self):
        self.frame_counter = 0
        last_frame = time.time()

        while True:
            read_sockets, write_sockets, exception_sockets = select.select(
                self.sockets_list, self.sockets_list[1:], self.sockets_list
            )
            for notified_socket in read_sockets:
                if notified_socket == self.server_socket:
                    try:
                        client_socket, client_address = self.server_socket.accept()
                        message = client_socket.recv(self.read_buffer)

                        if not len(message):
                            continue

                        self.sockets_list.append(client_socket)
                        print(
                            f"Accepted new connection from {client_address[0]}: {client_address[1]}!"
                        )
                        client_socket.send(self.feedback_msg)

                    except ConnectionAbortedError:
                        continue

                    except ConnectionResetError:
                        continue

                else:
                    try:
                        message = notified_socket.recv(self.read_buffer)
                        if not len(message):
                            self.sockets_list.remove(notified_socket)
                            continue
                        if message.decode("utf-8") == self.stop_command:
                            if notified_socket in self.sockets_list:
                                self.sockets_list.remove(notified_socket)
                            continue

                    except ConnectionAbortedError:
                        if notified_socket in self.sockets_list:
                            self.sockets_list.remove(notified_socket)
                        continue

                    except ConnectionResetError:
                        if notified_socket in self.sockets_list:
                            self.sockets_list.remove(notified_socket)
                        continue

            while time.time() - last_frame < self.frame_time:
                continue

            if self.frame_counter * self.frame_len == self.emg_data.shape[1]:
                self.frame_counter = 0
            index1 = int(self.frame_counter * self.frame_len)
            index2 = int(self.frame_counter * self.frame_len + self.frame_len)
            data_chunk = self.emg_data[:, index1:index2]

            data_to_send = data_chunk.tobytes(order="F")
            for notified_socket in write_sockets:
                try:
                    notified_socket.send(data_to_send)
                    last_frame = time.time()
                    self.frame_counter += 1

                except ConnectionAbortedError:
                    if notified_socket in self.sockets_list:
                        self.sockets_list.remove(notified_socket)
                    continue

                except ConnectionResetError:
                    if notified_socket in self.sockets_list:
                        self.sockets_list.remove(notified_socket)
                    continue


if __name__ == "__main__":
    emg_socket = EMGSimulator(
        tcp_ip="localhost",
        tcp_port=31000,
        fread=32,
        emg_data=RealTimeDataset(
            data_path=Path(r"emg_day_one.pkl"),  # TODO: change path to your data
            task="kinematics_chosen",  # TODO: change task to you what you want to simulate
        ).emg,
    )
    emg_socket.open_connection()
    print("Server open for connections!")
    emg_socket.stream_control()
