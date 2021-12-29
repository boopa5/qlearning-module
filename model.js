export default class Model {
    constructor({
        learningRate = 0.1,
        epsilon = 0.1,
        discountFactor = 0.9,
        epsilonDecay = 0.99,
        qTable = {}
    } = {
        learningRate: 0.5,
        epsilon: 0.1,
        discountFactor: 0.9,
        epsilonDecay: 0.9,
        qTable: {}
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

    get rewardFunc() {
        return this._rewardFunc;
    }

    set rewardFunc(rewardFunc) {
        this._rewardFunc = rewardFunc;
    }

    get qTable() {
        return this._qTable;
    }

    set qTable(qTable) {
        this._qTable = qTable;
    }

    decayEpsilon() {
        this._epsilon *= this._epsilonDecay;
    }

    updateQValue(state, action, newState, newActions, reward) {
        let oldValue = this.getQValue(state, action, "old");
        console.log("newState: " + JSON.stringify(newState));
        console.log("newActions: " + JSON.stringify(newActions));
        let y = reward + this._discountFactor * this.getQValue(newState, this.argmax(a => this.getQValue(newState, a), newActions));

        this._qTable[JSON.stringify(state)][action] = oldValue + this._learningRate * (y - oldValue);
    }

    argmax(func, inputs) {
        let index = [0];
        for (let i = 1; i < inputs.length; i++) {
            if (func(inputs[i]) > func(inputs[index[0]])) {
                index = [i];
            } else if (func(inputs[i]) == func(inputs[index[0]])) {
                index.push(i);
            }
        }
        return inputs[index[Math.floor(Math.random() * index.length)]];
    }

    getAction(state, actions) {
        let action;
        if (Math.random() < this._epsilon) {
            action = actions[Math.floor(Math.random() * actions.length)];
        }
        action = this.argmax(a => this.getQValue(state, a), actions);

        return action;
    }

    getQValue(state, action) {
        state = JSON.stringify(state);

        if (this._qTable[state] === undefined) {
            this._qTable[state] = {};
            this._qTable[state][action] = 0;
        }
        if (this._qTable[state][action] === undefined) {
            this._qTable[state][action] = 0;
        }
        return this._qTable[state][action];
    }
}