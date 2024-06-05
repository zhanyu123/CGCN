import os
import time

from pathlib import Path
from pprint import pprint
import logging
import numpy as np
import torch
# from caffe2.quantization.server.observer_test import net
from torch.utils.data import DataLoader

from utils.commend_parser import CommendArg
from utils.process_data import *
from utils.data_set import *
from utils.rsgd import *
# import geoopt
# # import geoopt.optim.RiemannianAdam
from models import *


class ProcessedData(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()

        self.train_triplets, self.valid_triplets, self.test_triplets, self.num_ent, self.num_rels = self.load_dataset()
        self.triplets = process({'train_triplets': self.train_triplets, 'valid_triplets': self.valid_triplets,
                                 'test_triplets': self.test_triplets}, self.num_rels)

        self.data_iter = self.get_data_iter()

        self.model = get_model(self.p, self.num_ent, self.num_rels)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}

    def load_dataset(self):
        dir = 'datasets'
        dir = os.path.join(dir, self.p.dataset)

        entity_path = os.path.join(dir, 'entities.dict')
        relation_path = os.path.join(dir, 'relations.dict')
        train_path = os.path.join(dir, 'train.txt')
        valid_path = os.path.join(dir, 'valid.txt')
        test_path = os.path.join(dir, 'test.txt')

        entity_dict = read_dictionary(entity_path)
        relation_dict = read_dictionary(relation_path)

        self.train_triplets = np.asarray(read_triplets(train_path, entity_dict,
                                                       relation_dict))
        self.valid_triplets = np.asarray(read_triplets(valid_path, entity_dict, relation_dict))
        self.test_triplets = np.asarray(read_triplets(test_path, entity_dict, relation_dict))
        self.num_nodes = len(entity_dict)
        self.num_rels = len(relation_dict)

        print("# entities: {}".format(self.num_nodes))
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.train_triplets)))
        


        return self.train_triplets, self.valid_triplets, self.test_triplets, self.num_nodes, self.num_rels

    def log(self):
        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.prj_path / 'logs' / self.p.name),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        pprint(vars(self.p))

    def best(self):

        save_root = self.prj_path / 'checkpoints'

        if not save_root.exists():
            save_root.mkdir()
        save_path = save_root / (self.p.name + '.pt')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train()
            val_results = self.evaluate('valid_triplets')

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val_results = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
            print(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
        self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')

        start = time.time()
        test_results = self.evaluate('test_triplets')
        end = time.time()
        self.logger.info(
            f"MRR: Tail {test_results['left_mrr']:.5}, Head {test_results['right_mrr']:.5}, Avg {test_results['mrr']:.5}")
        self.logger.info(
            f"MR: Tail {test_results['left_mr']:.5}, Head {test_results['right_mr']:.5}, Avg {test_results['mr']:.5}")
        self.logger.info(f"hits@1 = {test_results['hits@1']:.5}")
        self.logger.info(f"hits@3 = {test_results['hits@3']:.5}")
        self.logger.info(f"hits@10 = {test_results['hits@10']:.5}")
        self.logger.info("time ={}".format(end - start))

    def load_model(self, path):
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_val_mrr = self.best_val_results['mrr']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])


    def save_model(self, path):
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, path)
    def get_data_iter(self):

            def get_data_loader(dataset_class, split):
                return DataLoader(
                    dataset_class(self.triplets[split], self.num_ent, self.p),
                    batch_size=self.p.batch_size,
                    shuffle=True,
                    num_workers=self.p.num_workers,
                    pin_memory = True
                )
    
            return {
                'train_triplets': get_data_loader(TrainDataset, 'train_triplets'),
                'valid_triplets_head': get_data_loader(TestDataset, 'valid_triplets_head'),
                'valid_triplets_tail': get_data_loader(TestDataset, 'valid_triplets_tail'),
                'test_triplets_head': get_data_loader(TestDataset, 'test_triplets_head'),
                'test_triplets_tail': get_data_loader(TestDataset, 'test_triplets_tail')
            }

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train_triplets']
        for step, (triplets, labels) in enumerate(train_iter):
            triplets, labels = triplets.to(self.p.device), labels.to(self.p.device)
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred = self.model(subj, rel)
            loss = self.model.calc_loss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        loss = np.mean(losses)
        return loss

    

    def evaluate(self, split):
        def get_combined_results(left, right):
            results = dict()
            assert left['count'] == right['count']
            count = float(left['count'])
            results['left_mr'] = round(left['mr'] / count, 5)
            results['left_mrr'] = round(left['mrr'] / count, 5)
            results['right_mr'] = round(right['mr'] / count, 5)
            results['right_mrr'] = round(right['mrr'] / count, 5)
            results['mr'] = round((left['mr'] + right['mr']) / (2 * count), 5)
            results['mrr'] = round((left['mrr'] + right['mrr']) / (2 * count), 5)
            for k in [1, 3, 10]:
                results[f'left_hits@{k}'] = round(left[f'hits@{k}'] / count, 5)
                results[f'right_hits@{k}'] = round(right[f'hits@{k}'] / count, 5)
                results[f'hits@{k}'] = round((results[f'left_hits@{k}'] + results[f'right_hits@{k}']) / 2, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        right_result = self.predict(split, 'head')
        res = get_combined_results(left_result, right_result)
        return res


    def predict(self, split, mode):
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels) in enumerate(test_iter):
                triplets, labels = triplets.to(self.p.device), labels.to(self.p.device)
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                pred = self.model(subj, rel)
                b_range = torch.arange(pred.shape[0], device=self.p.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(labels.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred

                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0)

                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(ranks[ranks <= k]) + results.get(f'hits@{k}', 0)
        return results


if __name__ == '__main__':
    parser = CommendArg()
    args = parser.get_parser()
    data = ProcessedData(args)

    data.log()
    data.best()


