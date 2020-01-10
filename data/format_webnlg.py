from nltk import word_tokenize
import json, re, unidecode


DELIM = u"￨"

class Instance():
    def __init__(self, raw_instance):
        self.inputdict = self.tripleset2entities(raw_instance['modifiedtripleset'])
        self.lexicalizations = [
            word_tokenize(unidecode.unidecode(lex['lex']).lower())
            for lex in raw_instance['lexicalisations']]
        
        self.inputs, self.table = self.dict2inputsNtable()
        
    def dict2inputsNtable(self):
        """Transform the inputdict into onmt readable input"""
        res = list()
        table = list()
        for entidx, (entname, entlist) in enumerate(self.inputdict.items(), 1):
            entname = word_tokenize(entname)
            for idx, token in enumerate(entname, 1):
                input = DELIM.join([token, 
                                    'EntName',
                                    f'ENT{entidx}',
                                    str(idx),
                                    str(len(entname)+1-idx)])
                res.append(input)

            for value, key in entlist:
                value = word_tokenize(value)
                table.append((entname, [key], value))
                for idx, token in enumerate(value, 1):
                    input = DELIM.join([token,
                                        key,
                                        f'ENT{entidx}',
                                        str(idx),
                                        str(len(value)+1-idx)])
                    res.append(input)
        return res, table

    def tripleset2entities(self, tripleset):
        """Reads the raw triplets and do stuff"""
        ret = dict()
        for triple in tripleset:
            obj = self.clean_obj(triple['object'])
            prp = self.clean_prp(triple['property'])
            sbj = self.clean_obj(triple['subject'])

            if prp =='ethnicgroup':
                obj = obj.split('_in_')[0]
                obj = obj.split('_of_')[0]

            ret.setdefault(sbj, list())
            ret[sbj].append((obj, prp))
        return ret
    
    @staticmethod
    def clean_obj(s, lc=True):
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('_', ' ', s)  # turn undescores to spaces
        return s
    
    @staticmethod
    def clean_prp(s, lc=True):
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('\s+', '_', s)  # turn spaces to underscores
        s = re.sub('\s+\(in metres\)', '_m', s)
        s = re.sub('\s+\(in feet\)', '_f', s)
        s = re.sub('\(.*\)', '', s)
        return s.strip()    
    
def main():
    """We first deal with train and valid and then test"""
    for setname in ['train']:
    
        print(f"Starting with {setname}")
        path = f'webnlg/{setname}.json'
        with open(path, mode='r', encoding='utf8') as f:
            data = [item[str(idx+1)]
                    for idx, item in enumerate(json.load(f)['entries'])]

        dataset = [Instance(d) for d in data]
        input_filename = f'webnlg/{setname}_input.txt'
        output_filename = f'webnlg/{setname}_output.txt'
        table_filename = f'webnlg/{setname}_tables.jl'
        with open(input_filename, mode='w', encoding='utf8') as inputf:
            with open(table_filename, mode='w', encoding='utf8') as tablef:
                with open(output_filename, mode='w', encoding='utf8') as outputf:
                    for instance in dataset:
                        for lex in instance.lexicalizations:
                            inputf.write(' '.join(instance.inputs) + '\n')
                            tablef.write(json.dumps(instance.table) + '\n')
                            outputf.write(' '.join(lex) + '\n')
    
    for setname in ['test', 'valid']:
        print(f"Starting with {setname}")
        with open(f'webnlg/{setname}.json', mode='r', encoding='utf8') as f:
            data = [item[str(idx+1)]
                    for idx, item in enumerate(json.load(f)['entries'])]
        dataset = [Instance(d) for d in data]
        with open(f'webnlg/{setname}_input.txt', mode='w', encoding='utf8') as inputf:
            with open(f'webnlg/{setname}_tables.jl', mode='w', encoding='utf8') as tablef:
                with open(f'webnlg/{setname}_output.txt', mode='w', encoding='utf8') as outputf:
                    for instance in dataset:
                        inputf.write(' '.join(instance.inputs) + '\n')
                        tablef.write(json.dumps(instance.table) + '\n')
                        for lex in instance.lexicalizations:
                            outputf.write(' '.join(lex) + '\n')
                        outputf.write('\n')