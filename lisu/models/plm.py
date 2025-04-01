from pathlib import Path
import atexit

import numpy as np


class PseudoLabelManager(object):
    def __init__(self, save_path: Path, update_interval: list[int] | int):
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval
        self.updated_epochs = []

    def do_update(self, epoch):
        if isinstance(self.update_interval, int):
            if epoch not in self.updated_epochs and epoch % self.update_interval == 0:
                self.updated_epochs.append(epoch)
                return True
            return False
        else:
            raise NotImplementedError(
                f"Not implemented do_update check for type {type(self.update_interval)}"
            )

    def update(self, pred, batch_idx, frame_id, coord=None):
        assert pred.size(0) == batch_idx.size(0)
        assert pred.size(1) == 3
        assert len(frame_id) == batch_idx.max() + 1
        for b in range(int(batch_idx.max() + 1)):
            mask = batch_idx == b
            b_pred = pred[mask]
            file_path = self.save_path / (frame_id[b] + ".npy")
            file_path.unlink(missing_ok=True)
            with open(file_path, "wb") as f:
                np.save(f, b_pred.float().cpu().numpy())

    def configure_callbacks(self):
        atexit.register(self._clenup)

    def _clenup(self):
        if not self.save_path.is_dir():
            return
        for gt in self.save_path.iterdir():
            gt.unlink(missing_ok=True)
        self.save_path.rmdir()
