class Model {
    constructor({
        learningRate = 0.5,
        epsilon = 0.1,
        discountFactor = 0.9,
        epsilonDecay = 0.9,
        qTable = {}
    }) {
        this._learningRate = learningRate;
        this._epsilon = epsilon;
        this._discountFactor = discountFactor;
        this._epsilonDecay = epsilonDecay;
        this._qTable = qTable;
    }

    get learningRate() {
        return this._learningRate;
    }

    set learningRate(learningRate) {
        this._learningRate = learningRate;
    }

    get epsilon() {
       return this._epsilon;
    }

    set epsilon(epsilon) {
        this._epsilon = epsilon;
    }

    get discountFactor() {
        return this._discountFactor;
    }

    set discountFactor(discountFactor) {
        this._discountFactor = discountFactor;
    }

    get epsilonDecay() {
        return this._epsilonDecay;
    }

    get qTable() {
        return this._qTable;
    }

    set qTable(qTable) {
        this._qTable = qTable;
    }

    updateQValue(state, action, newState, newAction, reward) {
        let oldValue = this.getQValue(state, action);
        
        this._qTable[state][action] = newValue;
    }

    argmax(func, inputs) {
        let index = 0;
        for (let i = 0; i < inputs.length; i++) {
            if (func(inputs[i]) > func(inputs[index])) {
                index = i;
            }
        }
        return inputs[index];
    }

    getAction(state, actions) {
        let action;
        if (Math.random < this._epsilon) {
            action = actions[Math.floor(Math.random() * actions.length)];
        }
        action = this.argmax(a => this.getQValue(state, a), actions);
        this._epsilon *= this._epsilonDecay;
        return action;
    }

    getQValue(state, action) {
        if (this._qTable[state][action] === undefined) {
            this._qTable[state][action] = 0;
        }
        return this._qTable[state][action];
    }
}

export default Model;