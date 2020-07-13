import data_organise


def refine_X_Y():
    with open('./data_text_files/data_set.txt', 'r', encoding='utf-8') as stream:
        data = stream.read()
        person_lst = eval(data)
        stream.close()

    raw_X = []
    raw_Y = []

    for person in person_lst:
        raw_X.append(person['Resume']['Text'])
        if 'Rates' in person['Categories']['Business Line'] and 'Quantitative Analyst' in person['Categories']['Position Function']:
            raw_Y.append(1)
        else:
            raw_Y.append(0)

    X, Y = data_organise.remove_empty_rows(raw_X, raw_Y)

    return X, Y


if __name__ == '__main__':
    X, Y = refine_X_Y()
    print(len(X))
    print(len(Y))
