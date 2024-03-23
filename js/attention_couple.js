import { app } from "/scripts/app.js";

app.registerExtension({
	name: "AttentionCouple|cgem156",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AttentionCouple|cgem156") {
			const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				const r = origGetExtraMenuOptions?.apply?.(this, arguments);
                    options.unshift(
                        {
                            content: "add input",
                            callback: () => {
                                var index = 1;
                                if (this.inputs != undefined){
                                    index += this.inputs.length;
                                }
                                this.addInput("cond_" + Math.floor(index / 2), "CONDITIONING");
                                this.addInput("mask_" + Math.floor(index / 2), "MASK");
                            },
                        },
                        {
                            content: "remove input",
                            callback: () => {
                                if (this.inputs != undefined){
                                    this.removeInput(this.inputs.length - 1);
                                    this.removeInput(this.inputs.length - 1);
                                }								
                            },
                        },
                    );
                return r;
                
            }
        }
    },
});